# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A rule to expand LLVM target enumerations.

Replaces in a text file a single variable of the style `@LLVM_ENUM_FOOS@` with a
list of macro invocations, one for each target on its own line:

```
LLVM_FOO(TARGET1)
LLVM_FOO(TARGET2)
// ...
```

Example:
load(":enum_targets_gen.bzl", "enum_targets_gen")

enum_targets_gen(
    name = "disassemblers_def_gen",
    src = "include/llvm/Config/Disassemblers.def.in",
    out = "include/llvm/Config/Disassemblers.def",
    macro_name = "DISASSEMBLER",
    targets = llvm_target_disassemblers,
)

This rule provides a slightly more semantic API than template_rule, but the main
reason it exists is to permit a list with selects to be passed for `targets` as
a select is not allowed to be passed to a rule within another data structure. 
"""

def enum_targets_gen_impl(ctx):
    to_replace = ctx.attr.placeholder_name
    if not to_replace:
        to_replace = "@LLVM_ENUM_{}S@".format(ctx.attr.macro_name)
    replacement = "\n".join([
        "LLVM_{}({})\n".format(ctx.attr.macro_name, t)
        for t in ctx.attr.targets
    ])

    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = {to_replace: replacement},
    )

enum_targets_gen = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "out": attr.output(mandatory = True),
        "targets": attr.string_list(mandatory = True),
        "macro_name": attr.string(
            mandatory = True,
            doc = "The name of the enumeration. This is the suffix of the" +
                  " placeholder being replaced `@LLVM_ENUM_{}S@` and of the" +
                  " macro invocations generated `LLVM_{}(TARGET)`. Should be" +
                  " all caps and singular, e.g. 'DISASSEMBLER'",
        ),
        "placeholder_name": attr.string(
            doc = "The name of the placeholder. If unset, this defaults to" +
                  " `@LLVM_ENUM_{macro_name}S@`",
        ),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = enum_targets_gen_impl,
)
