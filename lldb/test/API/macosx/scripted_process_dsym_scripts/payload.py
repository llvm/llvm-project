import lldb

# The records live on the lldb module, which isn't reloaded with this script.


def __lldb_init_module(debugger, internal_dict):
    lldb.payload_inits = getattr(lldb, "payload_inits", 0) + 1


def __lldb_module_added_to_target(target, internal_dict):
    index = target.GetDebugger().GetIndexOfTarget(target)
    lldb.payload_targets = getattr(lldb, "payload_targets", []) + [index]
