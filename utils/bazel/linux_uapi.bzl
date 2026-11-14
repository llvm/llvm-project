# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extension for making Linux kernel UAPI headers available in Bazel."""

_SYSTEM_HEADERS_PATH_ENV_VAR = "LINUX_UAPI_INCLUDE_DIR"

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

def _setup_from_system(repository_ctx, include_dir):
    """Sets up the repository from a user-provided include directory."""
    if not include_dir.startswith("/"):
        fail("{} must be absolute, was {}".format(
            _SYSTEM_HEADERS_PATH_ENV_VAR,
            include_dir,
        ))

    if not repository_ctx.path(include_dir + "/linux").exists:
        fail(
            "{} does not contain Linux UAPI headers (no linux/ subdirectory)"
                .format(include_dir),
        )

    repository_ctx.symlink(include_dir, "include")
    repository_ctx.file("BUILD.bazel", _SYSTEM_BUILD)

    return None

_LIBC_DEV_PACKAGE_BUILD = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "linux_uapi_headers",
    # The linux-libc-dev package has headers for many platforms.
    # These may not compile, so use `textual_hdrs` instead of `hdrs`.
    textual_hdrs = glob(["usr/include/**/*.h"]),
    includes = [
        "usr/include",
    ] + select(
        {{
            # LLVM-libc requires the `asm` directory to exist,
            # so include the arch-specific directory.
            "@platforms//cpu:x86_64": ["usr/include/x86_64-linux-gnu"],
            "@platforms//cpu:arm64": ["usr/include/aarch64-linux-gnu"],
            "@platforms//cpu:riscv64": ["usr/include/riscv64-linux-gnu"],
        }},
        no_match_error = \"""Unsupported CPU for the Linux UAPI headers.

Use --repo_env={var_name}=/path/to/uapi/include to load from system headers,
or add support for your architecture.
\""",
    ),
    visibility = ["//visibility:public"],
)
""".format(var_name = _SYSTEM_HEADERS_PATH_ENV_VAR)

def _setup_from_libc_dev_package(repository_ctx):
    """Sets up linux UAPI headers from Debian's linux-libc-dev package."""

    # Pulling from Linux sources and generating the headers ourselves would
    # provide more control, but downloading from linux-libc-dev is much faster.
    repository_ctx.download_and_extract(
        url = "https://snapshot.debian.org/archive/debian/20260501T022911Z/pool/main/l/linux/linux-libc-dev_7.0.3-1_all.deb",
        sha256 = "02eb444940d2e8aee116b9b1525ca2618e48b025aff75e0b75515812af121c59",
    )
    repository_ctx.extract(archive = "data.tar.xz")
    repository_ctx.delete("data.tar.xz")
    repository_ctx.file("BUILD.bazel", _LIBC_DEV_PACKAGE_BUILD)

    return repository_ctx.repo_metadata(reproducible = True)

def _linux_uapi_setup_impl(repository_ctx):
    """Sets up a repository of Linux UAPI headers."""

    include_dir = repository_ctx.getenv(_SYSTEM_HEADERS_PATH_ENV_VAR, "")
    if include_dir:
        return _setup_from_system(repository_ctx, include_dir)

    return _setup_from_libc_dev_package(repository_ctx)

linux_uapi_setup = repository_rule(implementation = _linux_uapi_setup_impl)
