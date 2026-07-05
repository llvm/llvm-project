"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetName()
    obj.GetMangledName()
    obj.GetInstructions(lldb.SBTarget())
    sa = obj.GetStartAddress()
    ea = obj.GetEndAddress()
    # Do fuzz testing on the address obj, it should not crash lldb.
    import sb_address

    sb_address.fuzz_obj(sa)
    sb_address.fuzz_obj(ea)
    obj.GetPrologueByteSize
    obj.GetDescription(lldb.SBStream())
