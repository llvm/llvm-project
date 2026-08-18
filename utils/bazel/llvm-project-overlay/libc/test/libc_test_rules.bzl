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
load("//libc:libc_build_rules.bzl", "libc_common_copts", "libc_common_deps")
load("//libc:libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")

_TEST_DEFINES = ["LIBC_TEST_SUBPROCESS_TESTS=1"] + select({
    "//libc:full_build": ["TARGET_SUPPORTS_CLOCK"],
    "//conditions:default": [],
})

def libc_test(
        name,
        srcs = [],
        copts = [],
        deps = [],
        local_defines = [],
        linkopts = [],
        c_test = False,
        full_build_only = False,
        target_compatible_with = [],
        tags = [],
        **kwargs):
    """Add target for a libc test.

    Args:
      name: Test target name
      srcs: The list of sources for this test.
      copts: The list of options to add to the C++ compilation command.
      deps: The list of libc functions and libraries to be linked in.
      local_defines: The list of target local_defines if any.
      linkopts: Link options for the cc_test.
      c_test: Whether this test is a C unit test (uses LibcCTest).
      full_build_only: Whether the test should only be run in full-build mode.
      target_compatible_with: Constraints the target is compatible with.
      tags: Tags for the cc_test.
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
    ] + select({
        "//libc:full_build": [
            # Required by crt1.o. These would usually be provided by libc.a.
            "//libc:_r_debug",
            "//libc:environ",
            "//libc:program_invocation_name",
            "//libc:program_invocation_short_name",

            # Required by the hermetic test fixture or compiler codegen.
            "//libc/test:hermetic_test_codegen_deps",
        ],
        "//conditions:default": [],
    })

    if c_test:
        deps = deps + ["//libc/test/UnitTest:LibcCTest"]
    else:
        deps = deps + ["//libc/test/UnitTest:LibcUnitTest"]

    linkopts = linkopts + select({
        "//libc:full_build": [
            "-nolibc",
            "-nostartfiles",
            "-nostdlib++",
            "-static",
            "-Wl,-z,muldefs",
        ],
        "//conditions:default": [],
    })

    if full_build_only:
        target_compatible_with = target_compatible_with + select({
            "//libc:full_build": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })
        tags = tags + ["manual", "nobuildkite", "notap"]

    cc_test(
        name = name,
        srcs = srcs + select({
            "//libc:full_build": [
                "//libc/startup/linux:crt1",
            ],
            "//conditions:default": [],
        }),
        local_defines = local_defines + _TEST_DEFINES + LIBC_CONFIGURE_OPTIONS,
        deps = deps + libc_common_deps(),
        copts = copts + libc_common_copts(),
        linkopts = linkopts,
        linkstatic = 1,
        tags = tags,
        **kwargs
    )

def libc_test_library(name, copts = [], deps = [], local_defines = [], **kwargs):
    """Add target for library used in libc tests.

    Args:
      name: Library target name.
      copts: See cc_library.copts.
      deps: See cc_library.deps.
      local_defines: See cc_library.local_defines.
      **kwargs: Other attributes relevant to cc_library (e.g. "deps").
    """
    cc_library(
        name = name,
        testonly = True,
        copts = copts + libc_common_copts(),
        deps = deps + libc_common_deps(),
        local_defines = local_defines + _TEST_DEFINES + LIBC_CONFIGURE_OPTIONS,
        linkstatic = 1,
        **kwargs
    )
