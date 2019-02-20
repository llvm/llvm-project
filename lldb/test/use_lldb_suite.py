import inspect
import os
import sys


def find_lldb_root():
    lldb_root = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    while True:
        lldb_root = os.path.dirname(lldb_root)
        if lldb_root is None:
            return None

        test_path = os.path.join(lldb_root, "use_lldb_suite_root.py")
        if os.path.isfile(test_path):
            return lldb_root
    return None

lldb_root = find_lldb_root()
if lldb_root is not None:
    import imp
    fp, pathname, desc = imp.find_module("use_lldb_suite_root", [lldb_root])
    try:
        imp.load_module("use_lldb_suite_root", fp, pathname, desc)
    finally:
        if fp:
            fp.close()
