""" Test command for checking the Python commands can run in a stop-hook """
import lldb

did_run = False

class SomeCommand:
    def __init__(self, debugger, unused):
        self.dbg = debugger
    def __call__(self, debugger, command, exe_ctx, result):
        global did_run
        did_run = True
        result.PutCString("some output\n")

    def get_short_help(self):
        return "Test command - sets a variable."

class OtherCommand:
    def __init__(self, debugger, unused):
        self.dbg = debugger
    def __call__(self, debugger, command, exe_ctx, result):
        global did_run
        if did_run:
            result.SetStatus(lldb.eReturnStatusSuccessFinishNoResult)
        else:
            result.SetStatus(lldb.eReturnStatusFailed)

    def get_short_help(self):
        return "Test command - sets a variable."

def __lldb_init_module(debugger, unused):
    print("Adding command some-cmd and report-cmd")
    debugger.HandleCommand("command script add -c some_cmd.SomeCommand some-cmd")
    debugger.HandleCommand("command script add -c some_cmd.OtherCommand report-cmd")
