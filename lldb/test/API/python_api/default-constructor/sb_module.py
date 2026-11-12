"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetFileSpec()
    obj.GetPlatformFileSpec()
    obj.SetPlatformFileSpec(lldb.SBFileSpec())
    obj.GetUUIDString()
    obj.ResolveFileAddress(sys.maxsize)
    obj.ResolveSymbolContextForAddress(lldb.SBAddress(), 0)
    obj.GetDescription(lldb.SBStream())
    obj.GetNumSymbols()
    obj.GetSymbolAtIndex(sys.maxsize)
    sc_list = obj.FindFunctions("my_func")
    sc_list = obj.FindFunctions("my_func", lldb.eFunctionNameTypeAny)
    obj.FindGlobalVariables(lldb.SBTarget(), "my_global_var", 1)
    for section in obj.section_iter():
        s = str(section)
    for symbol in obj.symbol_in_section_iter(lldb.SBSection()):
        s = str(symbol)
    for symbol in obj:
        s = str(symbol)
    obj.GetAddressByteSize()
    obj.GetByteOrder()
    obj.GetTriple()
