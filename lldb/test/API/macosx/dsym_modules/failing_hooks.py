import lldb

# Count on the lldb module, which isn't reloaded with this script.


def __lldb_init_module(debugger, internal_dict):
    lldb.failing_hooks_inits = getattr(lldb, "failing_hooks_inits", 0) + 1
    raise RuntimeError("this script fails to load")
