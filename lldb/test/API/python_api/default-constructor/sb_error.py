"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetCString()
    obj.Fail()
    obj.Success()
    obj.GetError()
    obj.GetType()
    obj.SetError(5, lldb.eErrorTypeGeneric)
    obj.SetErrorToErrno()
    obj.SetErrorToGenericError()
    obj.SetErrorString("xyz")
    obj.SetErrorString(None)
    obj.SetErrorStringWithFormat("%s!", "error")
    obj.SetErrorStringWithFormat(None)
    obj.SetErrorStringWithFormat("error")
    obj.SetErrorStringWithFormat("%s %s", "warning", "danger")
    obj.SetErrorStringWithFormat("%s %s %s", "danger", "will", "robinson")
    obj.GetDescription(lldb.SBStream())
    obj.Clear()
