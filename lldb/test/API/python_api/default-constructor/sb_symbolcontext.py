"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetModule()
    obj.GetCompileUnit()
    obj.GetFunction()
    obj.GetBlock()
    obj.GetLineEntry()
    obj.GetSymbol()
    obj.GetDescription(lldb.SBStream())
