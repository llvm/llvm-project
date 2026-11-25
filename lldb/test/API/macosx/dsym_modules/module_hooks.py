import lldb

# Count on the lldb module, which isn't reloaded with this script.


def bump(name):
    setattr(lldb, name, getattr(lldb, name, 0) + 1)


def append(name, value):
    setattr(lldb, name, getattr(lldb, name, []) + [value])


def __lldb_init_module(debugger, internal_dict):
    bump("module_hooks_inits")
    append("module_hooks_debugger_ids", debugger.GetID())


def __lldb_module_added_to_target(target, internal_dict):
    bump("module_hooks_targets")
    append("module_hooks_added_targets", target)
