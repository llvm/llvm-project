import os, struct, signal

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread


class DummyStopHook:
    def __init__(self, target, args, internal_dict):
        self.target = target
        self.args = args

    def handle_stop(self, exe_ctx, stream):
        print("My DummyStopHook triggered. Printing args: \n%s" % self.args)
        sp = exe_ctx.process.GetScriptedImplementation()
        sp.handled_stop = True

class DummyScriptedProcess(ScriptedProcess):
    memory = None

    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.threads[0] = DummyScriptedThread(self, None)
        self.memory = {}
        addr = 0x500000000
        debugger = self.target.GetDebugger()
        index = debugger.GetIndexOfTarget(self.target)
        self.memory[addr] = "Hello, target " + str(index)
        self.handled_stop = False

    def read_memory_at_address(
        self, addr: int, size: int, error: lldb.SBError
    ) -> lldb.SBData:
        data = lldb.SBData().CreateDataFromCString(
            self.target.GetByteOrder(), self.target.GetCodeByteSize(), self.memory[addr]
        )

        return data

    def write_memory_at_address(self, addr, data, error):
        self.memory[addr] = data.GetString(error, 0)
        return len(self.memory[addr]) + 1

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

    def my_super_secret_method(self):
        if hasattr(self, "my_super_secret_member"):
            return self.my_super_secret_member
        else:
            return None


class DummyScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)
        self.frames.append({"pc": 0x0100001B00})

    def get_thread_id(self) -> int:
        return 0x19

    def get_name(self) -> str:
        return DummyScriptedThread.__name__ + ".thread-1"

    def get_state(self) -> int:
        return lldb.eStateStopped

    def get_stop_reason(self) -> Dict[str, Any]:
        return {"type": lldb.eStopReasonTrace, "data": {}}

    def get_register_context(self) -> str:
        return struct.pack(
            "21Q",
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        )


def __lldb_init_module(debugger, dict):
    # This is used when loading the script in an interactive debug session to
    # automatically, register the stop-hook and launch the scripted process.
    if not "SKIP_SCRIPTED_PROCESS_LAUNCH" in os.environ:
        debugger.HandleCommand(
            "target stop-hook add -k first -v 1 -k second -v 2 -P %s.%s"
            % (__name__, DummyStopHook.__name__)
        )
        debugger.HandleCommand(
            "process launch -C %s.%s" % (__name__, DummyScriptedProcess.__name__)
        )
    else:
        print(
            "Name of the class that will manage the scripted process: '%s.%s'"
            % (__name__, DummyScriptedProcess.__name__)
        )
        print(
            "Name of the class that will manage the stop-hook: '%s.%s'"
            % (__name__, DummyStopHook.__name__)
        )
