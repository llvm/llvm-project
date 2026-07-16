# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extension for making Linux kernel UAPI headers available in Bazel."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

_SYSTEM_HEADERS_PATH_ENV_VAR = "LINUX_UAPI_INCLUDE_DIR"

_HERMETIC_BUILD = """\
load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")
load("@rules_cc//cc:defs.bzl", "cc_library")

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = ["BUILD.bazel"],
    ),
)

make(
    name = "headers_gen",
    args = ["INSTALL_HDR_PATH=$$INSTALLDIR$$", "-j`nproc`"],
    lib_source = ":srcs",
    out_headers_only = True,
    targets = ["headers_install"],
)

cc_library(
    name = "linux_uapi_headers",
    hdrs = [":headers_gen"],
    includes = ["headers_gen/include"],
    visibility = ["//visibility:public"],
)
"""

def _linux_uapi_hermetic_setup(name):
    """Sets up a repository of UAPI headers directly from Linux sources."""

    git_repository(
        name = name,
        remote = "https://github.com/torvalds/linux.git",
        commit = "028ef9c96e96197026887c0f092424679298aae8",  # v7.0
        # Pull a minimal set of directories to avoid a several GiB download.
        sparse_checkout_patterns = [
            "arch/**",
            "include/uapi/**",
            "Makefile",
            "scripts/**",
        ],
        build_file_content = _HERMETIC_BUILD,
    )

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

_HERMETIC_ALIAS_BUILD = """\
alias(
    name = "linux_uapi_headers",
    actual = "@linux_uapi_hermetic//:linux_uapi_headers",
    visibility = ["//visibility:public"],
)
"""

def _linux_uapi_setup_impl(repository_ctx):
    """A repository of UAPI headers, either from a system directory or git."""
    include_dir = repository_ctx.os.environ.get(_SYSTEM_HEADERS_PATH_ENV_VAR, "")
    if include_dir:
        repository_ctx.symlink(include_dir, "include")
        repository_ctx.file("BUILD.bazel", _SYSTEM_BUILD)
    else:
        repository_ctx.file("BUILD.bazel", _HERMETIC_ALIAS_BUILD)

_linux_uapi_setup = repository_rule(
    implementation = _linux_uapi_setup_impl,
    environ = [_SYSTEM_HEADERS_PATH_ENV_VAR],
)

def _linux_uapi_extension_impl(_module_ctx):
    # Configure a separate hermetic repository, which will be conditionally
    # used based on the presence of _SYSTEM_HEADERS_PATH_ENV_VAR.
    #
    # Keeping this separate allows Bazel to skip fetching from git if unused.
    _linux_uapi_hermetic_setup(name = "linux_uapi_hermetic")

    # The actual repository to use, which may resolve to linux_uapi_hermetic.
    _linux_uapi_setup(name = "linux_uapi")

linux_uapi_extension = module_extension(
    implementation = _linux_uapi_extension_impl,
)
