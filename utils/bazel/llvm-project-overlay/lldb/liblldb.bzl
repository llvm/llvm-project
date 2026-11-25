# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for linking tools against liblldb.

With `--@llvm-project//llvm:link_llvm_dylib`, tools link the cc_shared_library
liblldb through `dynamic_deps`. Without it, they link the previous cc_binary
liblldb through `deps`, as they did before the cc_shared_library existed.
"""

load("//:vars.bzl", "PACKAGE_VERSION")

def liblldb_dynamic_deps():
    """Returns the `dynamic_deps` that link a tool against liblldb."""
    return select({
        Label("//llvm:link_llvm_dylib_enabled"): [
            Label("//lldb:lldb{}".format(PACKAGE_VERSION)),
        ],
        "//conditions:default": [],
    })

def liblldb_deps():
    """Returns the `deps` that link a tool against liblldb."""
    return select({
        Label("//llvm:link_llvm_dylib_enabled"): [],
        "//conditions:default": [Label("//lldb:liblldb.import")],
    })
