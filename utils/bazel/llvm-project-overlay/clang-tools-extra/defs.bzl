# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:paths.bzl", "paths")

def _symlink_impl(ctx):
    copied_files = []
    for input_file in ctx.files.srcs:
        (_, _, relative_filename) = input_file.path.rpartition(ctx.attr.partition)
        output_file = ctx.actions.declare_file(paths.join(ctx.attr.destination, relative_filename))
        ctx.actions.symlink(
            target_file = input_file,
            output = output_file,
        )
        copied_files.append(output_file)
    return DefaultInfo(files = depset(copied_files))

symlink = rule(
    implementation = _symlink_impl,
    attrs = {
        "destination": attr.string(mandatory = True),
        "partition": attr.string(mandatory = True),
        "srcs": attr.label_list(allow_files = True),
    },
)
