"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    try:
        obj.Initialize(lldb.SBDebugger.Create(), None, 0, "$", "^", True)
    except Exception:
        pass
    obj.IsActive()
    obj.IsDone()
    obj.SetIsDone(True)
    obj.GetGranularity()
