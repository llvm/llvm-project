import os,struct, signal

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread

class DummyScriptedProcess(ScriptedProcess):
    def __init__(self, exe_ctx: lldb.SBExecutionContext, args : lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.threads[0] = DummyScriptedThread(self, None)

    def get_memory_region_containing_address(self, addr: int) -> lldb.SBMemoryRegionInfo:
        return None

    def get_thread_with_id(self, tid: int):
        return {}

    def get_registers_for_thread(self, tid: int):
        return {}

    def read_memory_at_address(self, addr: int, size: int, error: lldb.SBError) -> lldb.SBData:
        data = lldb.SBData().CreateDataFromCString(
                                    self.target.GetByteOrder(),
                                    self.target.GetCodeByteSize(),
                                    "Hello, world!")

        return data

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def should_stop(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self):
        return DummyScriptedThread.__module__ + "." + DummyScriptedThread.__name__


class DummyScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)
        self.frames.append({"pc": 0x0100001b00 })

    def get_thread_id(self) -> int:
        return 0x19

    def get_name(self) -> str:
        return DummyScriptedThread.__name__ + ".thread-1"

    def get_state(self) -> int:
        return lldb.eStateStopped

    def get_stop_reason(self) -> Dict[str, Any]:
        return { "type": lldb.eStopReasonSignal, "data": {
            "signal": signal.SIGINT
        } }

    def get_register_context(self) -> str:
        return struct.pack(
                '21Q', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)


def __lldb_init_module(debugger, dict):
    if not 'SKIP_SCRIPTED_PROCESS_LAUNCH' in os.environ:
        debugger.HandleCommand(
            "process launch -C %s.%s" % (__name__,
                                     DummyScriptedProcess.__name__))
    else:
        print("Name of the class that will manage the scripted process: '%s.%s'"
                % (__name__, DummyScriptedProcess.__name__))
