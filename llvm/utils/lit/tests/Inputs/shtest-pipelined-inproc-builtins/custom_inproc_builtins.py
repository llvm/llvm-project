from __future__ import annotations

from lit.InprocBuiltins import InprocBuiltinIO
from lit.ShCommands import Command
from lit.ShellEnvironment import ShellEnvironment


def execute_print_out_err(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
) -> int:
    args = args[1:]

    if len(args) != 2:
        io.stderr.write("Expected two arguments.")
        return 1

    io.stdout.write(args[0])
    io.stderr.write(args[1])

    return 0


def execute_uppercaser(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
) -> int:
    io.stdout.write(io.stdin.read().upper())

    return 0


def execute_streq(
    cmd: Command, args: list[str], shenv: ShellEnvironment, io: InprocBuiltinIO
) -> int:
    args = args[1:]

    if len(args) != 1:
        io.stderr.write("Expected one argument.")
        return 2

    return int(io.stdin.read() != args[0])
