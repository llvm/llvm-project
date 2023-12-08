"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetName()
    obj.GetByteSize()
    # obj.GetEncoding(5)
    obj.GetNumberChildren(True)
    member = lldb.SBTypeMember()
    obj.GetChildAtIndex(True, 0, member)
    obj.GetChildIndexForName(True, "_member_field")
    obj.IsAPointerType()
    obj.GetPointeeType()
    obj.GetDescription(lldb.SBStream())
    obj.IsPointerType(None)
    lldb.SBType.IsPointerType(None)
    for child_type in obj:
        s = str(child_type)
