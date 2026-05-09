"""
dispatcher_service.py
---------------------
Full OOP orchestration pipeline for HealthRide auto-assignment.

Classes:
    DistanceCalculator  — GPS + Google Maps / Haversine distance logic
    DriverFilter        — All hard filters (availability, vehicle, time conflict)
    PostAssignHandler   — DB update, sync, notifications, logging
    DispatcherService   — Master orchestrator
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

from ai_assign_service import AIAssignService

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1 — DistanceCalculator
# ─────────────────────────────────────────────────────────────────────────────

class DistanceCalculator:
    """
    Calculates driving distance (miles) between driver GPS and pickup location.

    Priority:
        1. Google Maps Distance Matrix API  (real road distance)
        2. Haversine formula                (straight-line fallback)
        3. Stored distance_miles            (demo/mock fallback)
    """

    GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    EARTH_RADIUS_MILES = 3958.8

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    def _google_maps_distance(self, driver_gps: dict, pickup_address: str) -> Optional[float]:
        if not self.api_key:
            logger.warning("GOOGLE_MAPS_API_KEY not set in .env. Skipping Google Maps.")
            return None
        try:
            params = {
                "origins": f"{driver_gps['lat']},{driver_gps['lng']}",
                "destinations": pickup_address,
                "units": "imperial",
                "key": self.api_key,
            }
            response = requests.get(self.GOOGLE_MAPS_URL, params=params, timeout=5)
            element = response.json()["rows"][0]["elements"][0]

            if element["status"] != "OK":
                logger.warning(f"Google Maps status: {element['status']}")
                return None

            return round(element["distance"]["value"] / 1609.34, 2)

        except Exception as e:
            logger.error(f"Google Maps API error: {e}")
            return None

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return round(self.EARTH_RADIUS_MILES * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)

    def get_distance(self, driver: dict, pickup_address: str, pickup_gps: Optional[dict] = None) -> float:
        """
        Returns distance in miles from driver to pickup.
        Falls back gracefully through all 3 methods.
        """
        driver_gps = driver.get("last_gps")

        if driver_gps:
            distance = self._google_maps_distance(driver_gps, pickup_address)
            if distance is not None:
                logger.info(f"  [{driver['driver_id']}] Google Maps: {distance} miles")
                return distance

            if pickup_gps:
                distance = self._haversine_distance(
                    driver_gps["lat"], driver_gps["lng"],
                    pickup_gps["lat"], pickup_gps["lng"]
                )
                logger.info(f"  [{driver['driver_id']}] Haversine: {distance} miles")
                return distance

        fallback = driver.get("distance_miles", 0.0)
        logger.info(f"  [{driver['driver_id']}] Fallback: {fallback} miles")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 — DriverFilter
# ─────────────────────────────────────────────────────────────────────────────

class DriverFilter:
    """
    Applies all hard filters to eliminate invalid drivers for a trip.

    Filters:
        - Availability (on duty)
        - Vehicle type match
        - No scheduling time conflicts
    """

    def _is_available(self, driver: dict) -> bool:
        return driver.get("is_available", False)

    def _matches_vehicle(self, driver: dict, required_type: str) -> bool:
        if required_type == "standard":
            return True
        return required_type in driver.get("vehicle_types", [])

    def _has_time_conflict(self, driver: dict, trip_start: str, trip_end: str) -> bool:
        new_start = datetime.fromisoformat(trip_start)
        new_end = datetime.fromisoformat(trip_end)
        for slot in driver.get("scheduled_trips", []):
            s = datetime.fromisoformat(slot["start"])
            e = datetime.fromisoformat(slot["end"])
            if new_start < e and new_end > s:
                return True
        return False

    def filter(self, drivers: list[dict], trip: dict) -> list[dict]:
        """Runs all hard filters. Returns only valid candidate drivers."""
        required_type = trip.get("special_requirements", "standard")
        trip_start = trip["pickup_time"]
        trip_end = trip["approximate_dropoff_time"]

        logger.info(f"[Trip {trip['trip_id']}] Filtering {len(drivers)} drivers...")

        result = []
        for driver in drivers:
            if not self._is_available(driver):
                logger.info(f"  [{driver['driver_id']}] ✗ Not available")
                continue
            if not self._matches_vehicle(driver, required_type):
                logger.info(f"  [{driver['driver_id']}] ✗ Vehicle mismatch ({required_type})")
                continue
            if self._has_time_conflict(driver, trip_start, trip_end):
                logger.info(f"  [{driver['driver_id']}] ✗ Time conflict")
                continue
            logger.info(f"  [{driver['driver_id']}] ✓ Valid candidate")
            result.append(driver)

        logger.info(f"[Trip {trip['trip_id']}] {len(result)} valid candidates after filtering")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3 — PostAssignHandler
# ─────────────────────────────────────────────────────────────────────────────

class PostAssignHandler:
    """Handles all actions after a driver is selected."""

    def update_schedule_slot(self, trip_id: str, driver_id: str) -> None:
        logger.info(f"[DB] trip {trip_id} → driver {driver_id}")
        # TODO: db.execute("UPDATE trips SET driver_id=? WHERE trip_id=?", driver_id, trip_id)

    def sync_trip_with_driver(self, trip_id: str, driver_id: str) -> None:
        logger.info(f"[Sync] trip {trip_id} pushed to driver {driver_id}")
        # TODO: push_service.send(driver_id, trip_id)

    def log_decision(self, decision: dict) -> None:
        logger.info(f"[Log] {json.dumps(decision)}")
        # TODO: db.execute("INSERT INTO assignment_logs ...", decision)

    def trigger_notifications(self, trip_id: str, driver_id: str) -> None:
        logger.info(f"[Notify] SMS/push sent — trip {trip_id} → driver {driver_id}")
        # TODO: twilio.send(...) / firebase.notify(...)

    def run_all(self, trip_id: str, driver_id: str, decision: dict) -> None:
        self.update_schedule_slot(trip_id, driver_id)
        self.sync_trip_with_driver(trip_id, driver_id)
        self.log_decision(decision)
        self.trigger_notifications(trip_id, driver_id)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4 — DispatcherService (Master Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class DispatcherService:
    """
    Master orchestrator for the full auto-assignment pipeline.

    Pipeline:
        1. Filter drivers        (DriverFilter)
        2. Enrich with distance  (DistanceCalculator)
        3. AI selects driver     (AIAssignService)
        4. Post-assignment       (PostAssignHandler)
    """

    def __init__(self):
        self.filter = DriverFilter()
        self.distance_calculator = DistanceCalculator()
        self.ai_service = AIAssignService()
        self.post_assign = PostAssignHandler()

    def _enrich_with_distance(self, drivers: list[dict], trip: dict) -> list[dict]:
        pickup_address = trip["pickup_address"]
        pickup_gps = trip.get("pickup_gps")
        logger.info(f"[Trip {trip['trip_id']}] Calculating distances...")
        return [
            {**driver, "distance_miles": self.distance_calculator.get_distance(driver, pickup_address, pickup_gps)}
            for driver in drivers
        ]

    def auto_assign(self, trip: dict, all_drivers: list[dict]) -> Optional[dict]:
        """
        Runs the full auto-assignment pipeline for one trip.

        Args:
            trip:        Trip details dict
            all_drivers: All drivers from DB

        Returns:
            Assignment decision dict or None if failed.
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"AUTO-ASSIGN — Trip: {trip['trip_id']}")
        logger.info(f"{'='*55}")

        valid_drivers = self.filter.filter(all_drivers, trip)
        if not valid_drivers:
            logger.warning(f"[Trip {trip['trip_id']}] No valid drivers. Trip unassigned.")
            return None

        valid_drivers = self._enrich_with_distance(valid_drivers, trip)

        decision = self.ai_service.assign(trip, valid_drivers)
        if not decision:
            logger.error(f"[Trip {trip['trip_id']}] AI assignment failed.")
            return None

        self.post_assign.run_all(trip["trip_id"], decision["selected_driver_id"], decision)

        logger.info(f"\n✅ DONE — Trip {trip['trip_id']} → Driver {decision['selected_driver_id']}")
        logger.info(f"   Score: {decision['score']} | {decision['reasoning']}")

        return decision


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — VS Code Terminal Test Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("demo_data.json", "r") as f:
        data = json.load(f)

    trip = data["trip"]
    drivers = data["drivers"]

    print("\n── Input Trip ──")
    print(json.dumps(trip, indent=2))

    dispatcher = DispatcherService()
    result = dispatcher.auto_assign(trip, drivers)

    print("\n── Final Result ──")
    print(json.dumps(result, indent=2) if result else "No assignment made.")