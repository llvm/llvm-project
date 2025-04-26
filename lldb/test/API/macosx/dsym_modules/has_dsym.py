import lldb


def report_command(debugger, command, exe_ctx, result, internal_dict):
    result.AppendMessage(
        f'{lldb.num_module_inits} {lldb.num_target_inits} "{lldb.target_name}"'
    )
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)


def __lldb_init_module(debugger, internal_dict):
    # We only want to make one copy of the report command so it will be shared
    if "has_dsym_1" in __name__:
        # lldb is a convenient place to store our counters.
        lldb.num_module_inits = 0
        lldb.num_target_inits = 0
        lldb.target_name = "<unknown>"

        debugger.HandleCommand(
            f"command script add -o -f '{__name__}.report_command' report_command"
        )

    lldb.num_module_inits += 1


def __lldb_module_added_to_target(target, internal_dict):
    lldb.num_target_inits += 1
    target_name = target.executable.fullpath
