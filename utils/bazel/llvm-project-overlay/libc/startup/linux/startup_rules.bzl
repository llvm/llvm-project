# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for building startup objects."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _get_object_files(deps):
    """Gets object files that are directly provided by a target in deps.

    Args:
        deps: Targets from which to extract object files.

    Returns:
        Tuple of (objects, pic_objects) provided by some target in deps.
    """
    objects = []
    pic_objects = []
    for dep in deps:
        if CcInfo not in dep:
            fail("CcInfo not found in dep {}".format(dep.label))

        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            if linker_input.owner != dep.label:
                continue  # Only interested in directly owned linker inputs.

            for lib in linker_input.libraries:
                objects.extend(lib.objects or [])
                pic_objects.extend(lib.pic_objects or [])

    return objects, pic_objects

def _libc_startup_object_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".o")
    objects, pic_objects = _get_object_files([ctx.attr.dep])

    # Prefer nopic, as LLVM-libc currently only supports static non-PIE linking.
    # However, if only PIC objects are available, use them. This may happen
    # in toolchains where --force_pic is active (like llvm-project).
    #
    # This will likely need to change when we support Scrt1.o or rcrt1.o.
    input_objs = objects or pic_objects
    if len(input_objs) != 1:
        fail("Expected exactly one input object, got: {}".format(input_objs))

    input_obj = input_objs[0]

    ctx.actions.symlink(
        output = output,
        target_file = input_obj,
    )

    return [DefaultInfo(files = depset([output]))]

libc_startup_object = rule(
    implementation = _libc_startup_object_impl,
    attrs = {
        "dep": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
    },
)

def _filter_flags(
        flags,
        separate_flag_names,
        joined_flag_prefixes):
    """Filters flags to those in joined_flag_prefixes or separate_flag_names.

    Args:
        flags: The flags to filter.
        separate_flag_names: Names of flags whose value is specified separately
            from the flag (for example "--target value").
        joined_flag_prefixes: Prefixes of flags whose value is joined to the
            flag name (for example, --target=value).
    """
    filtered_flags = []
    skip_next = False
    for i, flag in enumerate(flags):
        if skip_next:
            skip_next = False
            continue

        if flag in separate_flag_names:
            if i + 1 < len(flags):
                filtered_flags.append(flag)
                filtered_flags.append(flags[i + 1])
                skip_next = True
            continue

        for prefix in joined_flag_prefixes:
            if flag.startswith(prefix):
                filtered_flags.append(flag)
                continue

    return filtered_flags

def _indirect_deps_linking_context(deps):
    """Creates a linking context with indirect inputs from deps."""
    direct_dep_labels = set([dep.label for dep in deps])
    indirect_dep_linker_inputs = [
        linker_input
        for dep in deps
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list()
        if linker_input.owner not in direct_dep_labels
    ]
    return cc_common.create_linking_context(
        linker_inputs = depset(indirect_dep_linker_inputs),
    )

def _create_merged_relocatable_object(
        ctx,
        inputs,
        output,
        linker,
        link_flags,
        cc_toolchain):
    """Configures an action to execute a relocatable link."""
    args = ctx.actions.args()
    args.add_all(link_flags)

    bindir = paths.dirname(linker)
    if bindir:
        args.add("-B" + bindir)

    args.add("-r")
    args.add("-nostdlib")
    args.add("-o", output)
    args.add_all(inputs)

    ctx.actions.run(
        outputs = [output],
        inputs = depset(
            inputs,
            transitive = [cc_toolchain.all_files],
        ),
        executable = linker,
        arguments = [args],
        mnemonic = "MergeRelocatableObject",
        use_default_shell_env = True,
    )

def _merge_relocatable_object_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    link_variables = cc_common.create_link_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        is_linking_dynamic_library = False,
    )
    link_flags = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
        variables = link_variables,
    )
    linker = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
    )
    relocatable_link_flags = _filter_flags(
        link_flags,
        ["-target", "--target", "--sysroot", "-isysroot"],
        ["-fuse-ld=", "-m", "--target=", "--sysroot="],
    )

    objects, pic_objects = _get_object_files(ctx.attr.deps)
    merged_pic_object = None
    merged_object = None

    if objects:
        merged_object = ctx.actions.declare_file(ctx.label.name + ".o")
        _create_merged_relocatable_object(
            ctx,
            inputs = objects,
            output = merged_object,
            linker = linker,
            link_flags = relocatable_link_flags,
            cc_toolchain = cc_toolchain,
        )

    if pic_objects:
        merged_pic_object = ctx.actions.declare_file(ctx.label.name + ".pic.o")
        _create_merged_relocatable_object(
            ctx,
            inputs = pic_objects,
            output = merged_pic_object,
            linker = linker,
            link_flags = relocatable_link_flags,
            cc_toolchain = cc_toolchain,
        )

    linking_context, _ = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = cc_common.create_compilation_outputs(
            objects = depset([merged_object]) if merged_object else None,
            pic_objects = depset([merged_pic_object]) if merged_pic_object else None,
        ),
        linking_contexts = [
            # Propagate transitive linking contexts from deps.
            # We only propagate indirect transitive dependencies because
            # the merged object file will contain symbols from direct deps.
            _indirect_deps_linking_context(ctx.attr.deps),
        ],
    )
    files = depset([o for o in [merged_object, merged_pic_object] if o])
    return [
        DefaultInfo(files = files),
        CcInfo(linking_context = linking_context),
    ]

merge_relocatable_object = rule(
    implementation = _merge_relocatable_object_impl,
    doc = """Merges multiple object files into a single relocatable object file.

    This rule mimics CMake's `merge_relocatable_object`,
    running the toolchain's linker driver `-r -nostdlib` on all direct deps.
    """,
    attrs = {
        "deps": attr.label_list(
            mandatory = True,
            providers = [CcInfo],
            doc = "The list of cc targets whose object files should be merged.",
        ),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
    provides = [DefaultInfo, CcInfo],
)
