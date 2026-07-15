# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extension for making Linux kernel UAPI headers available in Bazel."""

_SYSTEM_HEADERS_PATH_ENV_VAR = "LINUX_UAPI_INCLUDE_DIR"

_ERROR_MESSAGE_BZL = """\
def error_message(name, message):
    fail(message)
"""

_ERROR_BUILD = """\
load(":error_message.bzl", "error_message")

error_message(
    name = "linux_uapi_headers",
    message = \"""System linux UAPI headers not found.

Pass --repo_env={var_name}=/path/to/your/linux/include to bazel build.
\"""
)
""".format(var_name = _SYSTEM_HEADERS_PATH_ENV_VAR)

_SYSTEM_BUILD = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

filegroup(
    name = "srcs",
    srcs = glob(["include/**"]),
)

cc_library(
    name = "linux_uapi_headers",
    hdrs = [":srcs"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

def _linux_uapi_setup_impl(repository_ctx):
    """Sets up a repository of UAPI headers from a system directory"""
    include_dir = repository_ctx.os.environ.get(_SYSTEM_HEADERS_PATH_ENV_VAR, "")
    if include_dir:
        repository_ctx.symlink(include_dir, "include")
        repository_ctx.file("BUILD.bazel", _SYSTEM_BUILD)
    else:
        repository_ctx.file("error_message.bzl", _ERROR_MESSAGE_BZL)
        repository_ctx.file("BUILD.bazel", _ERROR_BUILD)

linux_uapi_setup = repository_rule(
    implementation = _linux_uapi_setup_impl,
    environ = [_SYSTEM_HEADERS_PATH_ENV_VAR],
)
