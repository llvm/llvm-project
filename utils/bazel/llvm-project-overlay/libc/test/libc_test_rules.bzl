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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//libc:libc_build_rules.bzl", "libc_common_copts")
load("//libc:libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")

_FULL_BUILD_COPTS = [
    "-nostdlib++",
    "-nostdlib",
    "-DLIBC_FULL_BUILD",
    "-DLIBC_COPT_USE_C_ASSERT",
]

def libc_test(
        name,
        copts = [],
        deps = [],
        local_defines = [],
        use_test_framework = True,
        full_build = False,
        **kwargs):
    """Add target for a libc test.

    Args:
      name: Test target name
      copts: The list of options to add to the C++ compilation command.
      deps: The list of libc functions and libraries to be linked in.
      local_defines: The list of target local_defines if any.
      use_test_framework: Whether to use the libc unit test `main` function.
      full_build: Whether to compile with LIBC_FULL_BUILD and disallow
          use of system headers. This is useful for tests that include both
          LLVM libc headers and proxy headers to avoid conflicting definitions.
      **kwargs: Attributes relevant for a cc_test.
    """
    deps = deps + [
        "//libc:hdr_stdint_proxy",
        "//libc:__support_macros_config",
        "//libc:__support_libc_errno",
        "//libc:errno",
        "//libc:func_aligned_alloc",
        "//libc:func_free",
        "//libc:func_malloc",
        "//libc:func_realloc",
    ]
    if use_test_framework:
        deps = deps + ["//libc/test/UnitTest:LibcUnitTest"]

    if full_build:
        copts = copts + _FULL_BUILD_COPTS
    cc_test(
        name = name,
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        deps = deps,
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
    cc_library(
        name = name,
        testonly = True,
        copts = copts + libc_common_copts(),
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        linkstatic = 1,
        **kwargs
    )
