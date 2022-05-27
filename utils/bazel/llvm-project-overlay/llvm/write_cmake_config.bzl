# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Rule for transforming CMake config files.
#
# Typical usage:
#   load("/tools/build_rules/write_cmake_config", "write_cmake_config")
#   write_cmake_config(
#       name = "ExpandMyTemplate",
#       src = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions

def write_cmake_config_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.files._script[0])
    args.add(ctx.file.src)
    args.add_all(ctx.attr.values)
    args.add("-o", ctx.outputs.out)
    ctx.actions.run(
        mnemonic = "WriteCMakeConfig",
        executable = "python3",
        inputs = ctx.files._script + [ ctx.file.src ],
        outputs = [ctx.outputs.out],
        arguments = [args],
    )

write_cmake_config = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "out": attr.output(mandatory = True),
        "values": attr.string_list(mandatory = True),
	"_script": attr.label(default="//llvm:write_cmake_config")
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = write_cmake_config_impl,
)
