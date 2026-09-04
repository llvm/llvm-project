# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extension for making Linux kernel UAPI headers available in Bazel."""

_SYSTEM_HEADERS_PATH_ENV_VAR = "LINUX_UAPI_INCLUDE_DIR"

_ERROR_MESSAGE_BZL = """\
def _error_message_impl(ctx):
    fail(ctx.attr.message)

error_message = rule(
    implementation = _error_message_impl,
    attrs = {"message": attr.string()},
)
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

cc_library(
    name = "linux_uapi_headers",
    # Builds using /usr/include may have a lot of junk that does not cleanly
    # compile, so use `textual_hdrs` instead of `hdrs`.
    textual_hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

def _default_include_dir(repository_ctx):
    """Returns a default include dir if it looks to have Linux UAPI headers."""
    include_dir = "/usr/include"
    if not repository_ctx.path(include_dir + "/linux").exists:
        return None
    return include_dir

def _linux_uapi_setup_impl(repository_ctx):
    """Sets up a repository of UAPI headers from a system directory."""
    include_dir = repository_ctx.getenv(_SYSTEM_HEADERS_PATH_ENV_VAR, "")
    if not include_dir:
        include_dir = _default_include_dir(repository_ctx)

    # If the directory doesn't exist, then set up a dummy rule that just errors.
    # This defers the error message until an actual build,
    # rather than failing during repository setup.
    if not include_dir:
        repository_ctx.file("error_message.bzl", _ERROR_MESSAGE_BZL)
        repository_ctx.file("BUILD.bazel", _ERROR_BUILD)
        return

    if not include_dir.startswith("/"):
        fail("{} must be absolute, was {}".format(
            _SYSTEM_HEADERS_PATH_ENV_VAR,
            include_dir,
        ))

    repository_ctx.symlink(include_dir, "include")
    repository_ctx.file("BUILD.bazel", _SYSTEM_BUILD)

linux_uapi_setup = repository_rule(implementation = _linux_uapi_setup_impl)
