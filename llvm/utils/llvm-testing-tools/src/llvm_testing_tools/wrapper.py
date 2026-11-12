# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Wraps the binaries contained in the package."""

import importlib.resources
import os
import subprocess
import sys


def run_test_tool():
    base_path = importlib.resources.files("llvm_testing_tools").joinpath("binaries")
    binary_path = os.path.join(str(base_path), os.path.basename(sys.argv[0]))
    result = subprocess.run([binary_path] + sys.argv[1:], stdin=sys.stdin)
    sys.exit(result.returncode)
