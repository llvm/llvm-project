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
    obj.ResolveFileAddress(sys.maxint)
    obj.ResolveSymbolContextForAddress(lldb.SBAddress(), 0)
    obj.GetDescription(lldb.SBStream())
    obj.GetNumSymbols()
    obj.GetSymbolAtIndex(sys.maxint)
    sc_list = obj.FindFunctions("my_func")
    sc_list = obj.FindFunctions("my_func", lldb.eFunctionNameTypeAny)
    obj.FindGlobalVariables(lldb.SBTarget(), "my_global_var", 1)
    for section in obj.section_iter():
        print section
    for symbol in obj.symbol_in_section_iter(lldb.SBSection()):
        print symbol
    for symbol in obj:
        print symbol
    obj.GetAddressByteSize()
    obj.GetByteOrder()
    obj.GetTriple()
