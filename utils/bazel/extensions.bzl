# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extensions for llvm-project"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load(":linux_uapi.bzl", "linux_uapi_setup")
load(":vulkan_sdk.bzl", "vulkan_sdk_setup")

_PYYAML_CONTENT = """\
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for PyYAML)
    licenses = ["notice"],
)

py_library(
    name = "yaml",
    srcs = glob(["yaml/*.py"]),
)
"""

def _llvm_repos_extension_impl(module_ctx):
    if any([m.is_root and m.name == "llvm-project-overlay" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "../../",
        )

    vulkan_sdk_setup(name = "vulkan_sdk")
    linux_uapi_setup(name = "linux_uapi")

    http_archive(
        name = "pyyaml",
        url = "https://github.com/yaml/pyyaml/archive/refs/tags/5.1.zip",
        sha256 = "f0a35d7f282a6d6b1a4f3f3965ef5c124e30ed27a0088efb97c0977268fd671f",
        strip_prefix = "pyyaml-5.1/lib3",
        build_file_content = _PYYAML_CONTENT,
    )

llvm_repos_extension = module_extension(
    implementation = _llvm_repos_extension_impl,
)
