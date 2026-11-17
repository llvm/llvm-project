# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for linking tools against the LLVM shared library."""

def llvm_dylib_deps():
    """Returns the `dynamic_deps` that link a binary against libLLVM.

    Empty unless `--@llvm-project//llvm:link_llvm_dylib` is set, which mirrors
    CMake's LLVM_LINK_LLVM_DYLIB.
    """
    return select({
        Label("//llvm:link_llvm_dylib_enabled"): [Label("//llvm:LLVM")],
        "//conditions:default": [],
    })

def llvm_libxml2_exports():
    """Returns the `exports_filter` for a cc_shared_library that links libxml2.

    Without libLLVM.so, a source-built libxml2 is linked into the shared
    library, and has to be exported for its dependents to use it from there.
    cc_shared_library resolves each exports_filter string against its own
    label, so the label is resolved here, where this repository's mapping
    applies to @libxml2.
    """
    return select({
        Label("//third-party:llvm_libxml2_from_source"): [str(Label("@libxml2//:xml2"))],
        "//conditions:default": [],
    })
