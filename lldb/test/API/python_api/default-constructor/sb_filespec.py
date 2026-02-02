"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.Exists()
    obj.ResolveExecutableLocation()
    obj.GetFilename()
    obj.GetDirectory()
    obj.GetPath(1)
    obj.GetDescription(lldb.SBStream())
