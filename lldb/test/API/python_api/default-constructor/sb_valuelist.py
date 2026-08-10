"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.Append(lldb.SBValue())
    obj.GetSize()
    obj.GetValueAtIndex(100)
    obj.FindValueObjectByUID(200)
    for val in obj:
        s = str(val)
