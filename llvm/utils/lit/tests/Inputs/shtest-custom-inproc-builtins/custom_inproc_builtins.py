from __future__ import annotations

from lit.InprocBuiltins import InprocBuiltinIO
from lit.ShCommands import Command
from lit.ShellEnvironment import ShellEnvironment


def returns_0(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
):
    return 0


def returns_1(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
):
    return 1


def custom_echo(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
):
    io.stdout.write(args[1])
    return 0


def echo_to_stderr(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
):
    io.stderr.write(args[1])
    return 0
