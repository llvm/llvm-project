import lldb

# __lldb_module_added_to_target adds a stop hook that counts the stops it runs
# at, with the options the test picks. The test can have __lldb_init_module add
# it instead.


class CountStops:
    def __init__(self, target, extra_args, internal_dict):
        pass

    def handle_stop(self, exe_ctx, stream):
        lldb.initial_stop_hook_runs = getattr(lldb, "initial_stop_hook_runs", 0) + 1
        return True


def add_hook(debugger, target):
    options = getattr(lldb, "initial_stop_hook_options", "")
    result = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand(
        f"target stop-hook add {options} -P {__name__}.CountStops",
        lldb.SBExecutionContext(target),
        result,
    )
    lldb.initial_stop_hook_added = result.Succeeded()


def __lldb_init_module(debugger, internal_dict):
    if getattr(lldb, "initial_stop_hook_from_init", False):
        add_hook(debugger, debugger.GetSelectedTarget())


def __lldb_module_added_to_target(target, internal_dict):
    if not getattr(lldb, "initial_stop_hook_from_init", False):
        add_hook(target.GetDebugger(), target)
