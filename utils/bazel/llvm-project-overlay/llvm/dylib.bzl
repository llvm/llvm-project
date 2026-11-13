# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for linking tools against the LLVM shared library."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")

def llvm_dylib_deps():
    """Returns the `dynamic_deps` that link a binary against libLLVM.

    Empty unless `--@llvm-project//llvm:link_llvm_dylib` is set, which mirrors
    CMake's LLVM_LINK_LLVM_DYLIB.
    """
    return select({
        Label("//llvm:link_llvm_dylib_enabled"): [Label("//llvm:LLVM")],
        "//conditions:default": [],
    })

def _shared_libraries_impl(ctx):
    infos = [dep[CcSharedLibraryInfo] for dep in ctx.attr.deps]
    linker_inputs = [
        entry.linker_input
        for entry in depset(
            [info for info in infos],
            transitive = [info.dynamic_deps for info in infos],
        ).to_list()
    ]
    libraries = [
        library
        for linker_input in linker_inputs
        for library in linker_input.libraries
    ]
    return [
        DefaultInfo(runfiles = ctx.runfiles(files = [
            library.dynamic_library
            for library in libraries
            if library.dynamic_library
        ])),
        CcInfo(linking_context = cc_common.create_linking_context(
            linker_inputs = depset([cc_common.create_linker_input(
                owner = ctx.label,
                libraries = depset(libraries),
            )]),
        )),
    ]

shared_libraries = rule(
    implementation = _shared_libraries_impl,
    doc = """Links cc_shared_library targets, and those they depend on, as CcInfo.

    For the `deps` of a cc_import whose library was linked against them, so its
    dependents link and load them too.
    """,
    attrs = {
        "deps": attr.label_list(providers = [CcSharedLibraryInfo]),
    },
    provides = [CcInfo],
)
