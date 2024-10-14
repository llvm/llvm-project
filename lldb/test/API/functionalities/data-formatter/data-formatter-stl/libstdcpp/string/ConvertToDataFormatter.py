# coding=utf8
"""
Helper formmater to verify Std::String by created via SBData
"""

import lldb


class SyntheticFormatter:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        return 6

    def has_children(self):
        return True

    def get_child_at_index(self, index):
        name = None
        match index:
            case 0:
                name = "short_string"
            case 1:
                name = "long_string"
            case 2:
                name = "short_string_ptr"
            case 3:
                name = "long_string_ptr"
            case 4:
                name = "short_string_ref"
            case 5:
                name = "long_string_ref"
            case _:
                return None

        child = self.valobj.GetChildMemberWithName(name)
        valType = child.GetType()
        return self.valobj.CreateValueFromData(name, child.GetData(), valType)


def __lldb_init_module(debugger, dict):
    typeName = "string_container"
    debugger.HandleCommand(
        'type synthetic add -x "'
        + typeName
        + '" --python-class '
        + f"{__name__}.SyntheticFormatter"
    )
