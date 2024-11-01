# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for tests.

libc functions are created though the libc_build_rules.bzl:libc_function.
They come in two flavors:
 - the internal one that is scoped into the `__llvm_libc` namespace.
 - the libc one that is the regular C function.

When performing tests we make sure to always use the internal version.
"""

load("//libc:libc_build_rules.bzl", "INTERNAL_SUFFIX")

def libc_test(name, srcs, libc_function_deps, deps = [], **kwargs):
    """Add target for a libc test.

    Args:
      name: Test target name
      srcs: List of sources for the test.
      libc_function_deps: List of libc_function targets used by this test.
      deps: The list of other libraries to be linked in to the test target.
      **kwargs: Attributes relevant for a cc_test. For example, name, srcs.
    """
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = [d + INTERNAL_SUFFIX for d in libc_function_deps] + [
            "//libc:libc_root",
            "//libc/test/UnitTest:LibcUnitTest",
        ] + deps,
        features = ["-link_llvmlibc"],  # Do not link libllvmlibc.a
        **kwargs
    )
