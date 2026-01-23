# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Re-export a cc_library with added LLVM specific settings.

This re-exports the dependent libraries in a way that satisfies layering_check

cc_library_wrapper(
    name = "library_wrapper",
    deps = [
        "@example//:library",
    ],
    defines = [
        "LLVM_ENABLE_EXAMPLE=1",
    ],
)
"""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

visibility("private")

def _cc_library_wrapper_impl(ctx):
    all_cc_infos = [dep[CcInfo] for dep in ctx.attr.deps]
    if ctx.attr.defines:
        all_cc_infos.append(CcInfo(
            compilation_context = cc_common.create_compilation_context(
                defines = depset(ctx.attr.defines),
            ),
        ))

    return cc_common.merge_cc_infos(direct_cc_infos = all_cc_infos)

cc_library_wrapper = rule(
    implementation = _cc_library_wrapper_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "Dependencies to cc_library targets to re-export.",
            providers = [CcInfo],
        ),
        "defines": attr.string_list(
            doc = "Additional preprocessor definitions to add to all dependent targets.",
            default = [],
        ),
    },
    doc = "Re-export a cc_library with added LLVM specific settings.",
    provides = [CcInfo],
)
