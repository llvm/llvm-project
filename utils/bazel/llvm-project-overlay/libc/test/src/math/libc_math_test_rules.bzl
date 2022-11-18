# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//libc/test:libc_test_rules.bzl", "libc_test")

"""LLVM libc starlark rules for math tests.

This rule delegates testing to libc_test_rules.bzl:libc_test.
It adds common math dependencies.
"""

def math_test(name, hdrs = [], deps = [], **kwargs):
    """Add a target for the unittest of a math function.

    Args:
      name: The name of the function being tested.
      hdrs: List of headers to add.
      deps: The list of other libraries to be linked in to the test target.
      **kwargs: Attributes relevant for a cc_test. For example, name, srcs.
    """
    test_name = name + "_test"
    libc_test(
        name = test_name,
        srcs = [test_name + ".cpp"] + hdrs,
        libc_function_deps = ["//libc:" + name],
        deps = [
            "//libc:__support_fputil_basic_operations",
            "//libc:__support_builtin_wrappers",
            "//libc:__support_fputil_fenv_impl",
            "//libc:__support_fputil_float_properties",
            "//libc:__support_fputil_fp_bits",
            "//libc:__support_uint128",
            "//libc:__support_fputil_manipulation_functions",
            "//libc:__support_fputil_nearest_integer_operations",
            "//libc/utils/UnitTest:fp_test_helpers",
        ] + deps,
        **kwargs
    )
