import os

import lldb
from lldb.plugins.scripted_platform import ScriptedPlatform


class MyScriptedPlatform(ScriptedPlatform):
    def __init__(self, exe_ctx, args):
        super().__init__(exe_ctx, args)

        if args and args.GetType() == lldb.eStructuredDataTypeDictionary:
            processes = args.GetValueForKey("processes")
            for i in range(0, processes.GetSize()):
                proc_info = processes.GetItemAtIndex(i)
                proc = {}
                proc["name"] = proc_info.GetValueForKey("name").GetStringValue(42)
                proc["arch"] = proc_info.GetValueForKey("arch").GetStringValue(42)
                proc["pid"] = proc_info.GetValueForKey("pid").GetIntegerValue()
                proc["parent"] = proc_info.GetValueForKey("parent").GetIntegerValue()
                proc["uid"] = proc_info.GetValueForKey("uid").GetIntegerValue()
                proc["gid"] = proc_info.GetValueForKey("gid").GetIntegerValue()
                self.processes[proc["pid"]] = proc

    def list_processes(self):
        return self.processes

    def attach_to_process(self, attach_info, target, debugger, error):
        return None

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
