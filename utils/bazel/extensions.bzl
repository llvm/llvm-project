# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""bzlmod extensions for llvm-project"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load(":vulkan_sdk.bzl", "vulkan_sdk_setup")

def _llvm_repos_extension_impl(module_ctx):
    if any([m.is_root and m.name == "llvm-project-overlay" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "../../",
        )

    http_archive(
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    http_archive(
        name = "vulkan_headers",
        build_file = "@llvm-raw//utils/bazel/third_party_build:vulkan_headers.BUILD",
        sha256 = "19f491784ef0bc73caff877d11c96a48b946b5a1c805079d9006e3fbaa5c1895",
        strip_prefix = "Vulkan-Headers-9bd3f561bcee3f01d22912de10bb07ce4e23d378",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/9bd3f561bcee3f01d22912de10bb07ce4e23d378.tar.gz",
        ],
    )

    vulkan_sdk_setup(name = "vulkan_sdk_setup")

    http_archive(
        name = "gmp",
        urls = [
            "https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz",
            "https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz",
        ],
        build_file = "@llvm-raw//utils/bazel/third_party_build:gmp.BUILD",
        sha256 = "fd4829912cddd12f84181c3451cc752be224643e87fac497b69edddadc49b4f2",
        strip_prefix = "gmp-6.2.1",
    )

    http_archive(
        name = "mpfr",
        urls = [
            "https://www.mpfr.org/mpfr-current/mpfr-4.2.2.tar.gz",
        ],
        sha256 = "826cbb24610bd193f36fde172233fb8c009f3f5c2ad99f644d0dea2e16a20e42",
        strip_prefix = "mpfr-4.2.2",
        build_file = "@llvm-raw//utils/bazel/third_party_build:mpfr.BUILD",
    )

    http_archive(
        name = "mpc",
        urls = [
            "https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz",
        ],
        sha256 = "ab642492f5cf882b74aa0cb730cd410a81edcdbec895183ce930e706c1c759b8",
        strip_prefix = "mpc-1.3.1",
        build_file = "@llvm-raw//utils/bazel/third_party_build:mpc.BUILD",
    )

    http_archive(
        name = "pfm",
        urls = [
            "https://versaweb.dl.sourceforge.net/project/perfmon2/libpfm4/libpfm-4.13.0.tar.gz",
        ],
        sha256 = "d18b97764c755528c1051d376e33545d0eb60c6ebf85680436813fa5b04cc3d1",
        strip_prefix = "libpfm-4.13.0",
        build_file = "@llvm-raw//utils/bazel/third_party_build:pfm.BUILD",
    )

    http_archive(
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )

    http_archive(
        name = "pybind11",
        url = "https://github.com/pybind/pybind11/archive/v2.10.3.zip",
        sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
        strip_prefix = "pybind11-2.10.3",
        build_file = "@llvm-raw//utils/bazel/third_party_build:pybind.BUILD",
    )

    http_archive(
        name = "pyyaml",
        url = "https://github.com/yaml/pyyaml/archive/refs/tags/5.1.zip",
        sha256 = "f0a35d7f282a6d6b1a4f3f3965ef5c124e30ed27a0088efb97c0977268fd671f",
        strip_prefix = "pyyaml-5.1/lib3",
        build_file = "@llvm-raw//utils/bazel/third_party_build:pyyaml.BUILD",
    )

    # TODO: bump to robin-map-1.4.0
    http_archive(
        name = "robin_map",
        build_file = "@llvm-raw//utils/bazel/third_party_build:robin_map.BUILD",
        sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
        strip_prefix = "robin-map-1.3.0",
        url = "https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz",
    )

    http_archive(
        name = "nanobind",
        build_file = "@llvm-raw//utils/bazel/third_party_build:nanobind.BUILD",
        sha256 = "8ce3667dce3e64fc06bfb9b778b6f48731482362fb89a43da156632266cd5a90",
        strip_prefix = "nanobind-2.9.2",
        url = "https://github.com/wjakob/nanobind/archive/refs/tags/v2.9.2.tar.gz",
    )

llvm_repos_extension = module_extension(
    implementation = _llvm_repos_extension_impl,
)
