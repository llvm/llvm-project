import os


class MissingMethodsScriptedProcess:
    def __init__(self, exe_ctx, args):
        pass


def __lldb_init_module(debugger, dict):
    if not "SKIP_SCRIPTED_PROCESS_LAUNCH" in os.environ:
        debugger.HandleCommand(
            "process launch -C %s.%s"
            % (__name__, MissingMethodsScriptedProcess.__name__)
        )
    else:
        print(
            "Name of the class that will manage the scripted process: '%s.%s'"
            % (__name__, MissingMethodsScriptedProcess.__name__)
        )
