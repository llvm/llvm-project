import os

import lldb
from lldb.plugins.scripted_platform import ScriptedPlatform


class MyScriptedPlatform(ScriptedPlatform):
    def __init__(self, exe_ctx, args):
        self.processes = {}

        proc = {}
        proc["name"] = "a.out"
        proc["arch"] = "arm64-apple-macosx"
        proc["pid"] = 420
        proc["parent"] = 42
        proc["uid"] = 501
        proc["gid"] = 20
        self.processes[420] = proc

    def list_processes(self):
        return self.processes

    def get_process_info(self, pid):
        return self.processes[pid]

    def launch_process(self, launch_info):
        return lldb.SBError()

    def kill_process(self, pid):
        return lldb.SBError()


def __lldb_init_module(debugger, dict):
    if not "SKIP_SCRIPTED_PLATFORM_SELECT" in os.environ:
        debugger.HandleCommand(
            "platform select scripted-platform -C %s.%s"
            % (__name__, MyScriptedPlatform.__name__)
        )
    else:
        print(
            "Name of the class that will manage the scripted platform: '%s.%s'"
            % (__name__, MyScriptedPlatform.__name__)
        )
