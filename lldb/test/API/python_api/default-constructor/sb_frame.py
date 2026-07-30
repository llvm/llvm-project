"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetFrameID()
    obj.GetPC()
    obj.SetPC(0xFFFFFFFF)
    obj.GetSP()
    obj.GetFP()
    obj.GetPCAddress()
    obj.GetSymbolContext(0)
    obj.GetModule()
    obj.GetCompileUnit()
    obj.GetFunction()
    obj.GetSymbol()
    obj.GetBlock()
    obj.GetFunctionName()
    obj.IsInlined()
    obj.EvaluateExpression("x + y")
    obj.EvaluateExpression("x + y", lldb.eDynamicCanRunTarget)
    obj.GetFrameBlock()
    obj.GetLineEntry()
    obj.GetThread()
    obj.Disassemble()
    obj.GetVariables(True, True, True, True)
    obj.GetVariables(True, True, True, False, lldb.eDynamicCanRunTarget)
    obj.GetRegisters()
    obj.FindVariable("my_var")
    obj.FindVariable("my_var", lldb.eDynamicCanRunTarget)
    obj.FindValue("your_var", lldb.eValueTypeVariableGlobal)
    obj.FindValue("your_var", lldb.eValueTypeVariableStatic, lldb.eDynamicCanRunTarget)
    obj.GetDescription(lldb.SBStream())
    obj.Clear()
