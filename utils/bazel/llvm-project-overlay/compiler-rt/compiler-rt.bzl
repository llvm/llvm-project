# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark for building parts of compiler-rt.

Variables provide baseline information for how to build various parts of
compiler-rt. These can be used to generate non-Bazel builds of the library.

Rules and macros support building the relevant filegroups of source files.

TODO: Add macros that provide a convenient way to construct a Bazel target for
the Clang resource directory with builtins and crt files.
"""

_common_copts = [
    "-O3",
    "-fPIC",
    "-ffreestanding",
    "-std=c11",
]

crt_copts = _common_copts + [
    "-DCRT_HAS_INITFINI_ARRAY",
    "-DEH_USE_FRAME_REGISTRY",
    "-fno-lto",
]

builtins_copts = _common_copts + [
    "-fno-builtin",
    "-fomit-frame-pointer",
    "-fvisibility=hidden",
    "-Wno-missing-prototypes",
    "-Wno-unused-parameter",
]

def _get_rel_path(path_str):
    rel_path = path_str.rpartition("/lib/builtins/")[2]
    if rel_path == path_str:
        fail("Expected '/lib/builtins/' in path " + path_str)
    return rel_path

def _filtered_builtins_srcs_impl(ctx):
    """Implementation of filter_builtins_srcs rule."""

    # Build a map from generic file basename to list of overriding files.
    overrides = {}
    for f in ctx.files.srcs:
        rel_path = _get_rel_path(f.short_path)
        if "/" in rel_path:
            base_file = rel_path.rpartition("/")[2]
            if base_file.endswith(".S"):
                base_file = base_file.removesuffix(".S") + ".c"
            overrides[base_file] = True

    filtered_files = []
    for f in ctx.files.srcs:
        rel_path = _get_rel_path(f.short_path)
        if "/" not in rel_path:
            # This is a generic file. Check if it's overridden.
            if rel_path not in overrides:
                filtered_files.append(f)
        else:
            # This is an arch-specific file, include it.
            filtered_files.append(f)

    # Remove any textual sources from this list.
    filtered_files = [
        f
        for f in filtered_files
        if f.extension not in ["inc", "def"]
    ]

    return [DefaultInfo(files = depset(filtered_files))]

filtered_builtins_srcs = rule(
    implementation = _filtered_builtins_srcs_impl,
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
            doc = "Input files.",
        ),
    },
    doc = """Build a filtered filegroup of non-textual srcs for builtins.

    Accepts a filegroup whose files are in lib/builtins/, and produces a target
    behaving like a filegroup containing filtered files.

    This removes any textual source files (`.inc` or `.def`) from the input.

    It also replaces generic srcs that are overridden by architecture-specific
    sources. For example, given a list of sources from filegroup of the form:

    - `.../lib/builtins/file_0.c`
    - `.../lib/builtins/file_1.c`
    - `.../lib/builtins/file_2.c`
    - `.../lib/builtins/arch/file_0.c`
    - `.../lib/builtins/arch/file_1.S`

    It removes any source-file at the top level of lib/builtins/ (e.g.
    lib/builtins/file_0.c) that has a corresponding source-file in an arch
    directory (e.g. lib/builtins/arch/file_0.c or lib/builtins/arch/file_1.S),
    producing a list like:

    - `.../lib/builtins/file_2.c`
    - `.../lib/builtins/arch/file_0.c`
    - `.../lib/builtins/arch/file_1.S`

    This allows a target architecture to simply add a specialized file to the
    list of sources with the architecture prefix and have the specialized
    version override the generic version.
    """,
)

def _filtered_builtins_textual_srcs_impl(ctx):
    """Implementation of filter_builtins_textual_srcs rule."""

    filtered_files = [
        f
        for f in ctx.files.srcs
        if f.extension in ["inc", "def"]
    ]

    return [DefaultInfo(files = depset(filtered_files))]

filtered_builtins_textual_srcs = rule(
    implementation = _filtered_builtins_textual_srcs_impl,
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
            doc = "Input files.",
        ),
    },
    doc = """Build a filegroup of the textual srcs for builtins.

    Textual sources are those that can't be compiled directly and aren't
    recognized as header files by Bazel. The extensions recognized here are
    `.inc` and `.def`.
    """,
)

def make_filtered_builtins_srcs_groups(name, srcs):
    """Macro to expand both the non-textual and textual filtered srcs groups."""
    if not name.endswith("_srcs"):
        fail("This rule's name must end with _srcs")
    filtered_builtins_srcs(
        name = name,
        srcs = srcs,
    )
    filtered_builtins_textual_srcs(
        name = name.removesuffix("_srcs") + "_textual_srcs",
        srcs = srcs,
    )
