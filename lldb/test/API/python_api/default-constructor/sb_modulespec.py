"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj: lldb.SBModuleSpec):
    obj.IsValid()
    obj.GetFileSpec()
    obj.GetPlatformFileSpec()
    obj.SetPlatformFileSpec(lldb.SBFileSpec())
    obj.GetSymbolFileSpec()
    obj.SetSymbolFileSpec(lldb.SBFileSpec())
    obj.GetObjectName()
    obj.SetObjectName("some object")
    obj.GetTriple()
    obj.SetTriple("a triple")
    obj.GetUUIDBytes()
    obj.GetUUIDLength()
    obj.SetUUIDBytes("SOMEBYTES".encode())
    obj.GetObjectOffset()
    obj.SetObjectOffset(320)
    obj.GetObjectSize()
    obj.SetObjectSize(3290)
    obj.GetDescription(lldb.SBStream())
    obj.GetTarget()
    obj.SetTarget(lldb.SBTarget())
    obj.Clear()
