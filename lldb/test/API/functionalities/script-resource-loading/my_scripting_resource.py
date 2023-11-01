import sys, lldb


def test_script_resource_loading(debugger, command, exe_ctx, result, dict):
    if not exe_ctx.target.process.IsValid():
        result.SetError("invalid process")
    process = exe_ctx.target.process
    if not len(process):
        result.SetError("invalid thread count")


def __lldb_init_module(debugger, dict):
    debugger.HandleCommand(
        "command script add -o -f a_out.test_script_resource_loading test_script_resource_loading"
    )
