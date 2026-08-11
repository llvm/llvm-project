#!/usr/bin/env python3


class OperatingSystemPlugIn(object):
    """OS plugin that hides the process's first real thread behind a virtual one."""

    def __init__(self, process):
        self.process = process

    def create_thread(self, tid, context):
        return None

    def get_thread_info(self):
        return [
            {
                "tid": 0x111111111,
                "name": "virtual",
                "queue": "queue",
                "state": "stopped",
                "stop_reason": "none",
                "core": 0,
            }
        ]

    def get_register_info(self):
        return None

    def get_register_data(self, tid):
        return None
