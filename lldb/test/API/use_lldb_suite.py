import inspect
import os
import sys


def find_lldb_root():
    lldb_root = os.path.realpath(
        os.path.dirname(inspect.getfile(inspect.currentframe())))
    while True:
        parent = os.path.dirname(lldb_root)
        if parent == lldb_root: # dirname('/') == '/'
            raise Exception("use_lldb_suite_root.py not found")
        lldb_root = parent

        test_path = os.path.join(lldb_root, "use_lldb_suite_root.py")
        if os.path.isfile(test_path):
            return lldb_root

lldb_root = find_lldb_root()

import imp
fp, pathname, desc = imp.find_module("use_lldb_suite_root", [lldb_root])
try:
    imp.load_module("use_lldb_suite_root", fp, pathname, desc)
finally:
    if fp:
        fp.close()
