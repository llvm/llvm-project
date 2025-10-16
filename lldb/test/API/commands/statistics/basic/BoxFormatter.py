import lldb


def summary(valobj, dict):
    return f"[{valobj.GetChildAtIndex(0).GetValue()}]"


def __lldb_init_module(debugger, dict):
    typeName = "Box<.*$"
    debugger.HandleCommand(
        'type summary add -x "'
        + typeName
        + '" --python-function '
        + f"{__name__}.summary"
    )
