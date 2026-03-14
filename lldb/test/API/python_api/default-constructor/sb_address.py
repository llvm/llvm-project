"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetFileAddress()
    obj.GetLoadAddress(lldb.SBTarget())
    obj.SetLoadAddress(0xFFFF, lldb.SBTarget())
    obj.OffsetAddress(sys.maxsize)
    obj.GetDescription(lldb.SBStream())
    obj.GetSection()
    obj.GetSymbolContext(lldb.eSymbolContextEverything)
    obj.GetModule()
    obj.GetCompileUnit()
    obj.GetFunction()
    obj.GetBlock()
    obj.GetSymbol()
    obj.GetLineEntry()
    obj.Clear()
