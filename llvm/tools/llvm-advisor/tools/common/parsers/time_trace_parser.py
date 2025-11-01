# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import json
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, TraceEvent


class TimeTraceParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.TIME_TRACE)

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_file(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            data = json.loads(content)
            events = self._parse_trace_events(data.get("traceEvents", []))

            metadata = {
                "total_events": len(events),
                "file_size": self.get_file_size(file_path),
                "beginning_of_time": data.get("beginningOfTime"),
            }

            return self.create_parsed_file(file_path, events, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_large_file(self, file_path: str) -> ParsedFile:
        events = []
        try:
            # For large files, parse in streaming mode
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1024 * 1024)  # Read first 1MB
                if content.startswith('{"traceEvents":['):
                    # Try to extract just a sample of events
                    bracket_count = 0
                    event_start = content.find("[") + 1
                    event_end = event_start

                    for i, char in enumerate(content[event_start:], event_start):
                        if char == "{":
                            bracket_count += 1
                        elif char == "}":
                            bracket_count -= 1
                            if bracket_count == 0 and len(events) < 1000:
                                try:
                                    event_json = content[event_start : i + 1]
                                    event_data = json.loads(event_json)
                                    event = self._parse_trace_event(event_data)
                                    if event:
                                        events.append(event)
                                except:
                                    pass
                                # Find next event
                                comma_pos = content.find(",", i)
                                if comma_pos == -1:
                                    break
                                event_start = comma_pos + 1

            metadata = {
                "total_events": len(events),
                "file_size": self.get_file_size(file_path),
                "is_partial": True,
            }

            return self.create_parsed_file(file_path, events, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_trace_events(self, events: List[Dict[str, Any]]) -> List[TraceEvent]:
        parsed_events = []
        for event_data in events:
            event = self._parse_trace_event(event_data)
            if event:
                parsed_events.append(event)
        return parsed_events

    def _parse_trace_event(self, event_data: Dict[str, Any]) -> TraceEvent:
        try:
            return TraceEvent(
                name=event_data.get("name", ""),
                category=event_data.get("cat", ""),
                phase=event_data.get("ph", ""),
                timestamp=event_data.get("ts", 0),
                duration=event_data.get("dur"),
                pid=event_data.get("pid"),
                tid=event_data.get("tid"),
                args=event_data.get("args", {}),
            )
        except Exception:
            return None

    def get_flamegraph_data(self, events: List[TraceEvent]) -> Dict[str, Any]:
        """Convert trace events to chrome tracing format with proper stacking"""
        stacks = []

        # Filter for duration events and sort by timestamp
        duration_events = [e for e in events if e.duration and e.duration > 0]
        duration_events.sort(key=lambda x: x.timestamp)

        # Build proper call stacks using begin/end events
        call_stack = []
        stack_frames = []

        # Group events by thread for proper stacking
        thread_events = {}
        for event in duration_events:
            tid = event.tid or 0
            if tid not in thread_events:
                thread_events[tid] = []
            thread_events[tid].append(event)

        # Process each thread separately
        all_samples = []
        for tid, events_list in thread_events.items():
            samples = self._build_call_stacks(events_list, tid)
            all_samples.extend(samples)

        # Sort all samples by timestamp for time order view
        all_samples.sort(key=lambda x: x["timestamp"])

        # Calculate total duration for normalization
        if all_samples:
            min_time = min(s["timestamp"] for s in all_samples)
            max_time = max(s["timestamp"] + s["duration"] for s in all_samples)
            total_duration = max_time - min_time
        else:
            total_duration = 0

        return {
            "samples": all_samples,
            "total_duration": total_duration,
            "thread_count": len(thread_events),
            "total_events": len(all_samples),
        }

    def _build_call_stacks(
        self, events: List[TraceEvent], thread_id: int
    ) -> List[Dict]:
        """Build proper call stacks from trace events"""
        samples = []
        active_frames = []  # Stack of currently active frames

        for event in events:
            # Create frame info
            frame = {
                "name": event.name,
                "category": event.category,
                "timestamp": event.timestamp,
                "duration": event.duration,
                "thread_id": thread_id,
                "depth": 0,
                "args": event.args or {},
            }

            # Find proper depth by checking overlaps with active frames
            frame["depth"] = self._calculate_frame_depth(frame, active_frames)

            # Add frame to samples
            samples.append(frame)

            # Update active frames list
            self._update_active_frames(frame, active_frames)

        return samples

    def _calculate_frame_depth(self, frame: Dict, active_frames: List[Dict]) -> int:
        """Calculate the proper depth for a frame based on overlapping frames"""
        frame_start = frame["timestamp"]
        frame_end = frame["timestamp"] + frame["duration"]

        # Find the maximum depth of overlapping frames
        max_depth = 0
        for active in active_frames:
            active_start = active["timestamp"]
            active_end = active["timestamp"] + active["duration"]

            # Check if frames overlap
            if frame_start < active_end and frame_end > active_start:
                max_depth = max(max_depth, active["depth"] + 1)

        return max_depth

    def _update_active_frames(self, new_frame: Dict, active_frames: List[Dict]):
        """Update the list of active frames, removing expired ones"""
        current_time = new_frame["timestamp"]

        # Remove frames that have ended
        active_frames[:] = [
            frame
            for frame in active_frames
            if frame["timestamp"] + frame["duration"] > current_time
        ]

        # Add new frame
        active_frames.append(new_frame)

    def get_sandwich_data(self, events: List[TraceEvent]) -> Dict[str, Any]:
        """Aggregate events by function name for sandwich view"""
        function_stats = {}

        duration_events = [e for e in events if e.duration and e.duration > 0]

        for event in duration_events:
            name = event.name
            if name not in function_stats:
                function_stats[name] = {
                    "name": name,
                    "total_time": 0,
                    "call_count": 0,
                    "category": event.category,
                    "avg_time": 0,
                }

            function_stats[name]["total_time"] += event.duration
            function_stats[name]["call_count"] += 1

        # Calculate averages and sort by total time
        for stats in function_stats.values():
            if stats["call_count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["call_count"]

        sorted_functions = sorted(
            function_stats.values(), key=lambda x: x["total_time"], reverse=True
        )

        return {"functions": sorted_functions}
