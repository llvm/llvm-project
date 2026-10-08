import lldb


def __lldb_init_module(debugger, internal_dict):
    lldb._dsym_codesign_test_loaded = True
