#
# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for license information.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//
#

import os
import subprocess
import sys


class ScriptError(Exception):
    """Convenience class for user errors generated"""

    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def error(msg):
    raise ScriptError(msg)


def print_line(msg, form="i"):
    print("{}: ({}) {}".format(os.path.basename(sys.argv[0]), form, msg))


def print_info_line(msg):
    print_line(msg)


def print_error_line(msg):
    print_line(msg, form="x")


class RunResult:
    """
    Auxiliary class for execute_command() containing the
    results of running a command
    """

    def __init__(self, args, stdout, stderr, returncode):
        self.executable = args[0]
        self.stdout = stdout.decode("utf-8")
        self.stderr = stderr.decode("utf-8")
        self.returncode = returncode
        self.command = " ".join(args)


def execute_command(args):
    """
    Run a command with arguments: args

    Return RunResult containing stdout, stderr, returncode
    """
    handle = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = handle.communicate()
    returncode = handle.wait()
    return RunResult(args, stdout, stderr, returncode)


# end of file
