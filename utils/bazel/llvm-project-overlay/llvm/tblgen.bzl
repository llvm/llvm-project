# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""This file contains BUILD extensions for generating source code from LLVM's
table definition files using the TableGen tool.

See http://llvm.org/cmds/tblgen.html for more information on the TableGen
tool.
TODO(chandlerc): Currently this expresses include-based dependencies as
"sources", and has no transitive understanding due to these files not being
correctly understood by the build system.
"""

def gentbl(
        name,
        tblgen,
        td_file,
        td_srcs,
        tbl_outs,
        library = True,
        tblgen_args = "",
        **kwargs):
    """gentbl() generates tabular code from a table definition file.

    Args:
      name: The name of the build rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      td_srcs: A list of table definition files included transitively.
      tbl_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to tblgen, and the out is the corresponding output file
        produced.
      library: Whether to bundle the generated files into a library.
      tblgen_args: Extra arguments string to pass to the tblgen binary.
      **kwargs: Keyword arguments to pass to subsidiary cc_library() rule.
    """
    llvm_project_execroot_path = Label("//llvm:tblgen.bzl").workspace_root

    if td_file not in td_srcs:
        td_srcs += [td_file]
    for (opts, out) in tbl_outs:
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))
        _gentbl(
            name = "%s_%s_rule" % (name, rule_suffix),
            tblgen = tblgen,
            td_file = td_file,
            td_srcs = td_srcs,
            opts = opts,
            out = out,
            tblgen_args = tblgen_args,
            llvm_project_execroot_path = llvm_project_execroot_path,
        )

    # For now, all generated files can be assumed to comprise public interfaces.
    # If this is not true, you should specify library = False
    # and list the generated '.inc' files in "srcs".
    if library:
        native.cc_library(
            name = name,
            # FIXME: This should be `textual_hdrs` instead of `hdrs`, but
            # unfortunately that doesn't work with `strip_include_prefix`:
            # https://github.com/bazelbuild/bazel/issues/12424
            #
            # Once that issue is fixed and released, we can switch this to
            # `textual_hdrs` and remove the feature disabling the various Bazel
            # features (both current and under-development) that motivated the
            # distinction between these two.
            hdrs = [f for (_, f) in tbl_outs],
            features = ["-parse_headers", "-header_modules"],
            **kwargs
        )

def _gentbl_impl(ctx):
    inputs = depset(ctx.files.td_srcs)

    args = ctx.actions.args()
    args.add("-I", "%s/llvm/include" % ctx.attr.llvm_project_execroot_path)
    args.add("-I", "%s/clang/include" % ctx.attr.llvm_project_execroot_path)
    args.add("-I", ctx.file.td_file.dirname)
    
    parsed_opts = ctx.attr.opts.split(' ')
    for opt in parsed_opts:
        args.add(opt)

    if ctx.attr.tblgen_args:
        parsed_args = ctx.attr.tblgen_args.split(' ')
        for tblgen_arg in parsed_args:
            args.add(tblgen_arg)

    args.add(ctx.file.td_file)
    args.add("-o", ctx.outputs.out)

    ctx.actions.run(
        mnemonic = "tblgen",
        executable = ctx.executable.tblgen,
        arguments = [args],
        inputs = inputs,
        outputs = [ctx.outputs.out],
    )

_gentbl = rule(
    implementation = _gentbl_impl,
    attrs = {
        "tblgen": attr.label(executable = True, cfg = "exec", doc = "The binary used to produce the output."),
        "td_file": attr.label(allow_single_file = True, doc = "The binary used to produce the output"),
        "td_srcs": attr.label_list(allow_files = True, doc = "A list of table definition files included transitively."),
        "opts": attr.string(doc = "String of options passed to tblgen."),
        "out": attr.output(doc = "Corresponding to opts output file."),
        "llvm_project_execroot_path": attr.string(doc = "Path to llvm-project execroot."),
        "tblgen_args": attr.string(default = "", doc = "Extra arguments string to pass to the tblgen binary."),
    }
)
