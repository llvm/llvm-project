# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository rule to statically link against the Vulkan SDK.

Requires installing the Vulkan SDK from https://vulkan.lunarg.com/.

If the Vulkan SDK is not installed, this generates an empty rule and you may
encounter linker errors like `error: undefined reference to 'vkCreateInstance'`.
"""

def _impl(repository_ctx):
    if "VULKAN_SDK" in repository_ctx.os.environ:
        sdk_path = repository_ctx.os.environ["VULKAN_SDK"]
        repository_ctx.symlink(sdk_path, "vulkan-sdk")

        repository_ctx.file("BUILD", """
cc_library(
    name = "sdk",
    srcs = select({
        "@bazel_tools//src/conditions:windows": [
            "vulkan-sdk/Lib/vulkan-1.lib"
        ],
        "//conditions:default": [
            "vulkan-sdk/lib/libvulkan.so.1",
        ],
    }),
    visibility = ["//visibility:public"],
)""")
    else:
        # Empty rule. Will fail to link for just targets that use Vulkan.
        repository_ctx.file("BUILD", """
cc_library(
    name = "sdk",
    srcs = [],
    visibility = ["//visibility:public"],
)""")

vulkan_sdk_setup = repository_rule(
    implementation = _impl,
    local = True,
)
