# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load(":vulkan_sdk.bzl", "vulkan_sdk_setup")
load(":configure.bzl", "llvm_configure", "DEFAULT_TARGETS")

def _llvm_configure_extension_impl(ctx):
    targets = []

    # Aggregate targets across imports.
    targets.extend([
        target
        for module in ctx.modules 
        for config in module.tags.configure 
        for target in config.targets 
        if target not in targets
    ])

    # Fall back to the default targets if all configurations of this extension
    # omit the `target` attribute.
    if targets == []:
        targets = DEFAULT_TARGETS

    llvm_configure(name = "llvm-project", targets = targets)

    # Deliberately omit the "llvm-raw" directory if we're not in the utils/bazel
    # directory.
    #
    # In downstream repos this intentionally causes the extension to immediately
    # error out if "llvm-raw" wasn't injected explicitly.
    #
    # We can't add this repo to the utils/bazel/MODULE.bazel as it would cause
    # submodule imports to resolve the new_local_repository at wrong paths.
    [
        new_local_repository(
            name = "llvm-raw",
            path = "../..",
            build_file_content = "# Empty."
        )
        for module in ctx.modules
        if module.name == "llvm-project-overlay" and module.is_root
    ]

    http_archive(
        name = "vulkan_headers",
        build_file = "@llvm-raw//utils/bazel/third_party_build:vulkan_headers.BUILD",
        sha256 = "19f491784ef0bc73caff877d11c96a48b946b5a1c805079d9006e3fbaa5c1895",
        strip_prefix = "Vulkan-Headers-9bd3f561bcee3f01d22912de10bb07ce4e23d378",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/9bd3f561bcee3f01d22912de10bb07ce4e23d378.tar.gz",
        ],
    )

    vulkan_sdk_setup(name = "vulkan_sdk")

    http_archive(
        name = "gmp",
        build_file = "@llvm-raw//utils/bazel/third_party_build:gmp.BUILD",
        sha256 = "fd4829912cddd12f84181c3451cc752be224643e87fac497b69edddadc49b4f2",
        strip_prefix = "gmp-6.2.1",
        urls = [
            "https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz",
            "https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz",
        ],
    )

    # https://www.mpfr.org/mpfr-current/
    #
    # When updating to a newer version, don't use URLs with "mpfr-current" in them.
    # Instead, find a stable URL like the one used currently.
    http_archive(
        name = "mpfr",
        build_file = "@llvm-raw//utils/bazel/third_party_build:mpfr.BUILD",
        sha256 = "9cbed5d0af0d9ed5e9f8dd013e17838eb15e1db9a6ae0d371d55d35f93a782a7",
        strip_prefix = "mpfr-4.1.1",
        urls = ["https://www.mpfr.org/mpfr-4.1.1/mpfr-4.1.1.tar.gz"],
    )

    http_archive(
        name = "pfm",
        build_file = "@llvm-raw//utils/bazel/third_party_build:pfm.BUILD",
        sha256 = "d18b97764c755528c1051d376e33545d0eb60c6ebf85680436813fa5b04cc3d1",
        strip_prefix = "libpfm-4.13.0",
        urls = ["https://versaweb.dl.sourceforge.net/project/perfmon2/libpfm4/libpfm-4.13.0.tar.gz"],
    )

    http_archive(
        name = "pybind11",
        build_file = "@llvm-raw//utils/bazel/third_party_build:pybind.BUILD",
        sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
        strip_prefix = "pybind11-2.10.3",
        url = "https://github.com/pybind/pybind11/archive/v2.10.3.zip",
    )

    # TODO: This exists in the BCR, but that implementation doesn't let us set
    #       `NB_BUILD=1` and `NB_SHARED=1`, so we use a custom variant here.
    #       Make `NB_BUILD` and `NB_SHARED` configurable upstream so that we can
    #       import this as a `bazel_dep` in `MODULE.bazel`.
    #
    #       See: https://github.com/bazelbuild/bazel-central-registry/blob/main/modules/nanobind/2.4.0/patches/add_build_file.patch
    http_archive(
        name = "nanobind",
        build_file = "@llvm-raw//utils/bazel/third_party_build:nanobind.BUILD",
        sha256 = "bb35deaed7efac5029ed1e33880a415638352f757d49207a8e6013fefb6c49a7",
        strip_prefix = "nanobind-2.4.0",
        url = "https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz",
    )


llvm_project_overlay = module_extension(
    doc = """Configure the llvm-project.

    Tags:
        targets: List of targets which Clang should support.
    """,
    implementation = _llvm_configure_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "targets": attr.string_list(),
            },
        ),
    },
)
