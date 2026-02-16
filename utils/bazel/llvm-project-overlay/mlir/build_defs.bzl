# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules and macros for MLIR"""

load("@rules_cc//cc:defs.bzl", "CcInfo", "cc_library")

def if_cuda_available(if_true, if_false = []):
    return select({
        # CUDA auto-detection is not yet supported.
        "//mlir:enable_cuda_config": if_true,
        "//conditions:default": if_false,
    })

def _cc_headers_only_impl(ctx):
    return CcInfo(compilation_context = ctx.attr.src[CcInfo].compilation_context)

cc_headers_only = rule(
    implementation = _cc_headers_only_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
    },
    doc = "Provides the headers from 'src' without linking anything.",
    provides = [CcInfo],
)

def mlir_c_api_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        header_deps = [],
        capi_deps = [],
        **kwargs):
    """Macro that generates three targets for MLIR C API libraries.

    * A standard cc_library target ("Name"),
    * A header-only cc_library target ("NameHeaders")
    * An implementation cc_library target tagged `alwayslink` suitable for
      inclusion in a shared library built with cc_binary() ("NameObjects").

    In order to avoid duplicate symbols, it is important that
    mlir_c_api_cc_library targets only depend on other mlir_c_api_cc_library
    targets via the "capi_deps" parameter. This makes it so that "FooObjects"
    depend on "BarObjects" targets and "Foo" targets depend on "Bar" targets.
    Don't cross the streams.
    """
    capi_header_deps = ["%sHeaders" % d for d in capi_deps]
    capi_object_deps = ["%sObjects" % d for d in capi_deps]
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + capi_deps + header_deps,
        **kwargs
    )
    cc_library(
        name = name + "Headers",
        hdrs = hdrs,
        deps = header_deps + capi_header_deps,
        **kwargs
    )
    cc_library(
        name = name + "Objects",
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + capi_object_deps + capi_header_deps + header_deps,
        alwayslink = True,
        **kwargs
    )

def nanobind_pyi_genrule(name, module_name, outs, deps, **kwargs):
    """Generates .pyi stub file(s) for a nanobind extension module.

    Args:
        name: Name of the generated target.
        module_name: Name of the module to generate stubs for (e.g., "_mlir").
        outs: List of expected output .pyi files.
        deps: All .so modules to load (including the target module).
        visibility: Visibility for the generated .pyi file(s).
    """
    deps_arg = ",".join(["$(location " + d + ")" for d in deps])

    native.genrule(
        name = name,
        srcs = deps,
        outs = outs,
        cmd = "$(location :stubgen_runner) --module " + module_name + " --deps " + deps_arg + " -o $(RULEDIR) > /dev/null",
        tools = [":stubgen_runner"],
        **kwargs
    )
