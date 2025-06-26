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

load("//libc:libc_build_rules.bzl", "libc_common_copts")
load("//libc:libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")

def libc_test(name, copts = [], deps = [], local_defines = [], **kwargs):
    """Add target for a libc test.

    Args:
      name: Test target name
      copts: The list of options to add to the C++ compilation command.
      deps: The list of libc functions and libraries to be linked in.
      local_defines: The list of target local_defines if any.
      **kwargs: Attributes relevant for a cc_test.
    """
    native.cc_test(
        name = name,
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        deps = [
            "//libc/test/UnitTest:LibcUnitTest",
            "//libc:__support_macros_config",
            "//libc:__support_libc_errno",
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
