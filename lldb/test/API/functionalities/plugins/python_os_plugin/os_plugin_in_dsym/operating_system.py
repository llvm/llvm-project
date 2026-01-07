#!/usr/bin/env python

import lldb
import struct

# Value is:
#   0 called - state is not stopped
#   1 called - state is stopped
#   2 not called

stop_state = {
    "in_init": 2,
    "in_get_thread_info": 2,
    "in_create_thread": 2,
    "in_get_register_info": 2,
    "in_get_register_data": 2,
}


def ReportCommand(debugger, command, exe_ctx, result, unused):
    global stop_state
    for state in stop_state:
        result.AppendMessage(f"{state}={stop_state[state]}\n")
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)


class OperatingSystemPlugIn:
    """This class checks that all the"""

    def __init__(self, process):
        """Initialization needs a valid.SBProcess object.
        global stop_state

        This plug-in will get created after a live process is valid and has stopped for the
        first time."""
        self.process = process
        stop_state["in_init"] = self.state_is_stopped()
        interp = process.target.debugger.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        cmd_str = (
            f"command script add test_report_command -o -f {__name__}.ReportCommand"
        )
        interp.HandleCommand(cmd_str, result)

    def state_is_stopped(self):
        if self.process.state == lldb.eStateStopped:
            return 1
        else:
            return 0

    def does_plugin_report_all_threads(self):
        return True

    def create_thread(self, tid, context):
        global stop_state
        stop_state["in_create_thread"] = self.state_is_stopped()

        return None

    def get_thread_info(self):
        global stop_state
        stop_state["in_get_thread_info"] = self.state_is_stopped()
        idx = self.process.threads[0].idx
        return [
            {
                "tid": 0x111111111,
                "name": "one",
                "queue": "queue1",
                "state": "stopped",
                "stop_reason": "breakpoint",
                "core": idx,
            }
        ]

    def get_register_info(self):
        global stop_state
        stop_state["in_get_register_info"] = self.state_is_stopped()
        return None

    def get_register_data(self, tid):
        global stop_state
        stop_state["in_get_register_data"] = self.state_is_stopped()
        return None
