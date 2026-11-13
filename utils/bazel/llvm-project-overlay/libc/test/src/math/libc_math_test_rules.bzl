# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for math tests.

This rule delegates testing to libc_test_rules.bzl:libc_test.
It adds common math dependencies.
"""

load("//libc/test:libc_test_rules.bzl", "libc_test")

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
        deps = ["//libc:func_name".replace("func_name", name)] + [
            "//libc/src/__support/CPP:algorithm",
            "//libc/src/__support/CPP:bit",
            "//libc/src/__support/CPP:limits",
            "//libc/src/__support/CPP:type_traits",
            "//libc/src/__support/FPUtil:basic_operations",
            "//libc/src/__support/FPUtil:cast",
            "//libc/src/__support/FPUtil:fenv_impl",
            "//libc/src/__support/FPUtil:fp_bits",
            "//libc/src/__support/FPUtil:manipulation_functions",
            "//libc/src/__support/FPUtil:nearest_integer_operations",
            "//libc/src/__support/FPUtil:normal_float",
            "//libc/src/__support/macros:optimization",
            "//libc/src/__support/macros/properties:architectures",
            "//libc/src/__support/macros/properties:cpu_features",
            "//libc/src/__support/macros/properties:os",
            "//libc/src/__support/macros/properties:types",
            "//libc/src/__support:math_extras",
            "//libc/src/__support:uint128",
            "//libc/hdr:errno_macros",
            "//libc/hdr:fenv_macros",
            "//libc/hdr:math_macros",
            "//libc/test/UnitTest:fp_test_helpers",
        ] + deps,
        **kwargs
    )

def math_mpfr_test(name, hdrs = [], deps = [], **kwargs):
    """Add a target for the unittest of a math function.

    Args:
      name: The name of the function being tested.
      hdrs: List of headers to add.
      deps: The list of other libraries to be linked in to the test target.
      **kwargs: Attributes relevant for a cc_test. For example, name, srcs.
    """
    math_test(
        name = name,
        hdrs = hdrs,
        overlay_build_only = True,  # mpfr cannot be built in full-build yet
        deps = deps + ["//libc/utils/MPFRWrapper:mpfr_wrapper"],
        **kwargs
    )
