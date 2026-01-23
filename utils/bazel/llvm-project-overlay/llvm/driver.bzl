# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration for the llvm-driver tool."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_cc//cc:defs.bzl", "CcInfo", "cc_binary")

# Mapping from every tool to the cc_library that implements the tool's entrypoint.
_TOOLS = {
    "clang-scan-deps": "//clang:clang-scan-deps-lib",
    "clang": "//clang:clang-driver",
    "dsymutil": "//llvm:dsymutil-lib",
    "lld": "//lld:lld-lib",
    "llvm-ar": "//llvm:llvm-ar-lib",
    "llvm-cgdata": "//llvm:llvm-cgdata-lib",
    "llvm-cxxfilt": "//llvm:llvm-cxxfilt-lib",
    "llvm-debuginfod-find": "//llvm:llvm-debuginfod-find-lib",
    "llvm-dwp": "//llvm:llvm-dwp-lib",
    "llvm-gsymutil": "//llvm:llvm-gsymutil-lib",
    "llvm-ifs": "//llvm:llvm-ifs-lib",
    "llvm-libtool-darwin": "//llvm:llvm-libtool-darwin-lib",
    "llvm-lipo": "//llvm:llvm-lipo-lib",
    "llvm-ml": "//llvm:llvm-ml-lib",
    "llvm-mt": "//llvm:llvm-mt-lib",
    "llvm-nm": "//llvm:llvm-nm-lib",
    "llvm-objcopy": "//llvm:llvm-objcopy-lib",
    "llvm-objdump": "//llvm:llvm-objdump-lib",
    "llvm-rc": "//llvm:llvm-rc-lib",
    "llvm-readobj": "//llvm:llvm-readobj-lib",
    "llvm-size": "//llvm:llvm-size-lib",
    "llvm-symbolizer": "//llvm:llvm-symbolizer-lib",
    "sancov": "//llvm:sancov-lib",
}

# Tools automatically get their own name as an alias, but there may be additional
# aliases for a given tool.
_EXTRA_ALIASES = {
    "clang": ["clang++", "clang-cl", "clang-cpp"],
    "lld": ["ld", "lld-link", "ld.lld", "ld64.lld", "wasm-ld"],
    "llvm-ar": ["ranlib", "lib", "dlltool"],
    "llvm-cxxfilt": ["c++filt"],
    "llvm-objcopy": ["bitcode-strip", "install-name-tool", "strip"],
    "llvm-objdump": ["otool"],
    "llvm-rc": ["windres"],
    "llvm-readobj": ["readelf"],
    "llvm-symbolizer": ["addr2line"],
}

def _validated_string_list_flag_impl(ctx):
    invalid_values = [v for v in ctx.build_setting_value if v not in ctx.attr.values]
    if invalid_values:
        fail("Tool(s) [{}] are not in the known list of tools: {}".format(
            ", ".join(invalid_values),
            ", ".join(ctx.attr.values),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

# Like string_list_flag, but with the validation that string_flag provides.
_validated_string_list_flag = rule(
    implementation = _validated_string_list_flag_impl,
    build_setting = config.string_list(flag = True),
    attrs = {
        "values": attr.string_list(
            doc = "The list of allowed values for this setting. An error is raised if any other value is given.",
        ),
    },
    doc = "A string list-typed build setting that can be set on the command line",
)

def generate_driver_selects(name):
    """Generates flags and config settings to configure the tool list.

    By default, all supported tools are included in the "llvm" driver binary.
    To build only a subset, specify just the subset you want as the flag.
    For example, to produce a binary with just llvm-nm and llvm-size, run:

        $ bazel build \
            --@llvm-project//llvm:driver-tools=llvm-nm,llvm-size \
            @llvm-project//llvm:llvm

    Note: this assumes the flag name is "driver-tools" by being invoked as:
        generate_driver_selects(name = "driver-tools")

    Args:
      name: the name of the flag that configures which tools are included.
    """

    _validated_string_list_flag(
        name = name,
        build_setting_default = _TOOLS.keys(),
        values = _TOOLS.keys(),
    )
    for tool in _TOOLS.keys():
        native.config_setting(
            name = "{}-include-{}".format(name, tool),
            flag_values = {name: tool},
        )

def select_driver_tools(flag):
    """Produce a list of tool deps based on generate_driver_selects().

    Args:
      flag: name that was used for generate_driver_selects().
    Returns:
      List of tool deps based on generate_driver_selects().
    """
    tools = []
    for tool, target in _TOOLS.items():
        tools += select({
            "{}-include-{}".format(flag, tool): [target],
            "//conditions:default": [],
        })
    return tools

def _generate_driver_tools_def_impl(ctx):
    # Depending on how the LLVM build files are included,
    # it may or may not have the @llvm-project repo prefix.
    # Compare just on the name. We could also include the package,
    # but the name itself is unique in practice.
    label_to_name = {Label(v).name: k for k, v in _TOOLS.items()}

    # Reverse sort by the *main* tool name, but keep aliases together.
    # This is consistent with how tools/llvm-driver/CMakeLists.txt does it,
    # and this makes sure that more specific tools are checked first.
    # For example, "clang-scan-deps" should not match "clang".
    tools = [label_to_name[tool.label.name] for tool in ctx.attr.driver_tools]
    tool_alias_pairs = []
    for tool_name in reversed(tools):
        tool_alias_pairs.append((tool_name, tool_name))
        for extra_alias in _EXTRA_ALIASES.get(tool_name, []):
            tool_alias_pairs.append((tool_name, extra_alias))

    lines = [
        'LLVM_DRIVER_TOOL("{alias}", {tool})'.format(
            tool = tool_name.replace("-", "_"),
            alias = alias.removeprefix("llvm-"),
        )
        for (tool_name, alias) in tool_alias_pairs
    ]
    lines.append("#undef LLVM_DRIVER_TOOL")

    ctx.actions.write(
        output = ctx.outputs.out,
        content = "\n".join(lines),
    )

generate_driver_tools_def = rule(
    implementation = _generate_driver_tools_def_impl,
    doc = """Generate a list of LLVM_DRIVER_TOOL macros.
See tools/llvm-driver/CMakeLists.txt for the reference implementation.""",
    attrs = {
        "driver_tools": attr.label_list(
            doc = "List of tools to include in the generated header. Use select_driver_tools() to provide this.",
            providers = [CcInfo],
        ),
        "out": attr.output(
            doc = "Name of the generated .def output file.",
            mandatory = True,
        ),
    },
)

def llvm_driver_cc_binary(
        name,
        deps = None,
        **kwargs):
    """cc_binary wrapper for binaries using the llvm-driver template."""
    expand_template(
        name = "_gen_" + name,
        out = name + "-driver.cpp",
        substitutions = {"@TOOL_NAME@": name.replace("-", "_")},
        template = "//llvm:cmake/modules/llvm-driver-template.cpp.in",
    )
    deps = deps or []
    cc_binary(
        name = name,
        srcs = [name + "-driver.cpp"],
        deps = deps + ["//llvm:Support"],
        **kwargs
    )
