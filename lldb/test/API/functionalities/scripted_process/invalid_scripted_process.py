import os, struct, signal

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread


class InvalidScriptedProcess(ScriptedProcess):
    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.threads[0] = InvalidScriptedThread(self, None)

    def read_memory_at_address(
        self, addr: int, size: int, error: lldb.SBError
    ) -> lldb.SBData:
        error.SetErrorString("This is an invalid scripted process!")
        return lldb.SBData()

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 666

    def should_stop(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self):
        return InvalidScriptedThread.__module__ + "." + InvalidScriptedThread.__name__


class InvalidScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)

    def get_thread_id(self) -> int:
        return 0x19

    def get_name(self) -> str:
        return InvalidScriptedThread.__name__ + ".thread-1"

    def get_state(self) -> int:
        return lldb.eStateInvalid

    def get_stop_reason(self) -> Dict[str, Any]:
        return {"type": lldb.eStopReasonSignal, "data": {"signal": signal.SIGTRAP}}

    def get_register_context(self) -> str:
        return None


def __lldb_init_module(debugger, dict):
    if not "SKIP_SCRIPTED_PROCESS_LAUNCH" in os.environ:
        debugger.HandleCommand(
            "process launch -C %s.%s" % (__name__, InvalidScriptedProcess.__name__)
        )
    else:
        print(
            "Name of the class that will manage the scripted process: '%s.%s'"
            % (__name__, InvalidScriptedProcess.__name__)
        )
