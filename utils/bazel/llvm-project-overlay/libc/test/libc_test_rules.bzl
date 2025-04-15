# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for tests.

libc functions are created though the libc_build_rules.bzl:libc_function.
They come in two flavors:
 - the internal one that is scoped into the `LIBC_NAMESPACE` namespace.
 - the libc one that is the regular C function.

When performing tests we make sure to always use the internal version.
"""

load("//libc:libc_build_rules.bzl", "libc_common_copts", "libc_internal_target")
load("//libc:libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")

def libc_test(name, srcs, libc_function_deps = [], copts = [], deps = [], local_defines = [], **kwargs):
    """Add target for a libc test.

    Args:
      name: Test target name
      srcs: List of sources for the test.
      libc_function_deps: List of libc_function targets used by this test.
      copts: The list of options to add to the C++ compilation command.
      deps: The list of other libraries to be linked in to the test target.
      local_defines: The list of target local_defines if any.
      **kwargs: Attributes relevant for a libc_test. For example, name, srcs.
    """
    native.cc_test(
        name = name,
        srcs = srcs,
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        deps = [libc_internal_target(d) for d in libc_function_deps] + [
            "//libc/test/UnitTest:LibcUnitTest",
            "//libc:__support_macros_config",
            "//libc:errno",
            "//libc:func_aligned_alloc",
            "//libc:func_free",
            "//libc:func_malloc",
            "//libc:func_realloc",
        ] + deps,
        copts = copts + libc_common_copts(),
        linkstatic = 1,
        **kwargs
    )

def libc_test_library(name, copts = [], local_defines = [], **kwargs):
    """Add target for library used in libc tests.

    Args:
      name: Library target name.
      copts: See cc_library.copts.
      local_defines: See cc_library.local_defines.
      **kwargs: Other attributes relevant to cc_library (e.g. "deps").
    """
    native.cc_library(
        name = name,
        testonly = True,
        copts = copts + libc_common_copts(),
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        linkstatic = 1,
        **kwargs
    )
