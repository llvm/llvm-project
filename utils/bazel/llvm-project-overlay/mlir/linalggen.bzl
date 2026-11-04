# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BUILD extensions for MLIR linalg generation."""

load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("@rules_cc//cc:defs.bzl", "cc_library")

def genlinalg(name, linalggen, src, linalg_outs):
    """genlinalg() generates code from a tc spec file.

    Args:
      name: The name of the build rule for use in dependencies.
      linalggen: The binary used to produce the output.
      src: The tc spec file.
      linalg_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to linalggen, and the out is the corresponding output file
        produced.
    """

    for (opts, out) in linalg_outs:
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))
        run_binary(
            name = "%s_%s_genrule" % (name, rule_suffix),
            srcs = [src],
            outs = [out],
            # `$@` in opts names the output, as it would in a genrule.
            args = [
                opt.replace("$@", "$(execpath %s)" % out)
                for opt in opts.split(" ")
            ] + ["$(execpath %s)" % src],
            tool = linalggen,
        )

    hdrs = [f for (opts, f) in linalg_outs]
    cc_library(
        name = name,
        hdrs = hdrs,
        textual_hdrs = hdrs,
    )
