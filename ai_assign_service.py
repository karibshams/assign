"""
ai_assign_service.py
--------------------
Handles all AI decision-making for driver assignment using OpenAI.

Score = (distance_miles * 0.6) + (current_load * 0.4)
Lower score = better driver.
"""

import json
import logging
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class AIAssignService:
    """
    Responsible for sending trip + candidate drivers to OpenAI
    and returning the best driver assignment decision.
    """

    MODEL = "gpt-4o"
    MAX_TOKENS = 1024
    TEMPERATURE = 0

    SYSTEM_PROMPT = """
You are a dispatcher AI for a medical transportation company.

Your job: select the BEST driver for a trip using this formula:
score = (distance_miles * 0.6) + (current_load * 0.4)

Rules:
- Pick the driver with the LOWEST score.
- On tie, prefer lower current_load.
- All drivers are pre-filtered and valid.

Respond ONLY with this JSON, nothing else:
{
  "trip_id": "<trip_id>",
  "selected_driver_id": "<driver_id>",
  "score": <float>,
  "reasoning": "<one sentence>"
}
"""

    def __init__(self):
        self.client = OpenAI()  # reads OPENAI_API_KEY from .env

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_payload(self, trip: dict, candidate_drivers: list[dict]) -> dict:
        return {
            "trip": {
                "trip_id": trip["trip_id"],
                "pickup_time": trip["pickup_time"],
                "approximate_dropoff_time": trip["approximate_dropoff_time"],
                "pickup_address": trip["pickup_address"],
                "dropoff_address": trip["dropoff_address"],
                "special_requirements": trip.get("special_requirements", "standard"),
            },
            "candidate_drivers": [
                {
                    "driver_id": d["driver_id"],
                    "driver_name": d["driver_name"],
                    "distance_miles": d["distance_miles"],
                    "current_load": d["current_load"],
                }
                for d in candidate_drivers
            ],
        }

    def _parse_response(self, raw_text: str, trip_id: str) -> Optional[dict]:
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"[Trip {trip_id}] JSON parse error: {e}")
            return None

        required = {"trip_id", "selected_driver_id", "score", "reasoning"}
        missing = required - decision.keys()
        if missing:
            logger.error(f"[Trip {trip_id}] Missing fields: {missing}")
            return None

        return decision

    def _call_api(self, user_message: str, trip_id: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Trip {trip_id}] OpenAI API error: {e}")
            return None

    # ── Public ────────────────────────────────────────────────────────────────

    def assign(self, trip: dict, candidate_drivers: list[dict]) -> Optional[dict]:
        """
        Main method. Sends trip + drivers to OpenAI.
        Returns assignment decision dict or None on failure.
        """
        trip_id = trip["trip_id"]

        if not candidate_drivers:
            logger.warning(f"[Trip {trip_id}] No candidates. Skipping AI call.")
            return None

        logger.info(f"[Trip {trip_id}] Calling OpenAI with {len(candidate_drivers)} candidates...")

        payload = self._build_payload(trip, candidate_drivers)
        user_message = (
            "Select the best driver for this trip.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

        raw_text = self._call_api(user_message, trip_id)
        if not raw_text:
            return None

        logger.info(f"[Trip {trip_id}] OpenAI response: {raw_text}")

        decision = self._parse_response(raw_text, trip_id)
        if decision:
            logger.info(
                f"[Trip {trip_id}] ✅ Selected: {decision['selected_driver_id']} "
                f"| Score: {decision['score']} | {decision['reasoning']}"
            )

        return decision