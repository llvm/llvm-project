from __future__ import annotations

import functools

import lit.util
from lit.InprocBuiltins import InprocBuiltin
from lit.LitConfig import LitConfig
from lit.llvm.daemon_tool import DaemonTool, invoke_llvm_daemon_tool
from lit.TestingConfig import TestingConfig


def check_daemon_support(executable: str) -> bool:
    """
    Check that the executable at the given executable path supports being run
    as an LLVM daemon tool, by testing that we receive the `ready` message
    when starting the daemon.
    """

    try:
        daemon = DaemonTool(executable)
        daemon.start_daemon()
        daemon.send_commands(["exit"])
    except:
        return False

    return True


def get_daemon_tool_inproc_builtin(
    tool_name: str,
    executable: str,
) -> InprocBuiltin:
    result = InprocBuiltin(
        functools.partial(invoke_llvm_daemon_tool, executable),
    )
    # Command which the in-process built-in falls back to in situations where
    # running as an in-process built-in doesn't make sense, or daemonization is
    # disabled for this tool.
    result.fallback_command = executable
    # Identifies this in-process built-in as the daemonized replacement for the
    # given tool name; used to implement DISALLOW_DAEMONIZATION: {tool}.
    result.llvm_daemon_tool_identifier = tool_name
    return result


def use_daemon_tool(
    testing_config: TestingConfig,
    lit_config: LitConfig,
    tool_dir: str,
    tool_name: str,
    inproc_builtin_key_override: str | None = None,
) -> bool:
    """
    Add the in-process built-in for running an LLVM tool in daemon mode.
    The key for the in-process built-in is the absolute path to the tool,
    so that it automatically replaces uses of the tool in testing, unless
    `inproc_builtin_key_override` is specified, in which case it will
    be the string provided via that parameter.
    """

    # Find the absolute path to the executable in the tool directory.
    executable = lit.util.which(tool_name, tool_dir)
    if not executable:
        lit_config.warning(
            f"Could not find executable for requested daemon tool "
            f"{tool_name} in {tool_dir}."
        )
        return False

    if not check_daemon_support(executable):
        lit_config.warning(
            f"Executable {executable} does not support running as an LLVM "
            f"daemon tool."
        )
        return False

    # Register an in-process built-in to run the tool in daemon mode. This
    # replaces invocations to the tool.
    key = inproc_builtin_key_override if inproc_builtin_key_override else executable
    testing_config.test_format.extra_inproc_builtins[
        key
    ] = get_daemon_tool_inproc_builtin(tool_name, executable)

    return True
