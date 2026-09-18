"""
This file contains the Bazel build dependencies for Google Benchmark (both C++ source and Python bindings).
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def benchmark_deps():
    """Loads dependencies required to build Google Benchmark."""

    if "bazel_skylib" not in native.existing_rules():
        http_archive(
            name = "bazel_skylib",
            sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
            urls = [
                "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
                "https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
            ],
        )

    if "rules_python" not in native.existing_rules():
        http_archive(
            name = "rules_python",
            sha256 = "e85ae30de33625a63eca7fc40a94fea845e641888e52f32b6beea91e8b1b2793",
            strip_prefix = "rules_python-0.27.1",
            url = "https://github.com/bazelbuild/rules_python/releases/download/0.27.1/rules_python-0.27.1.tar.gz",
        )

    if "com_google_googletest" not in native.existing_rules():
        new_git_repository(
            name = "com_google_googletest",
            remote = "https://github.com/google/googletest.git",
            tag = "release-1.12.1",
        )

    if "nanobind" not in native.existing_rules():
        new_git_repository(
            name = "nanobind",
            remote = "https://github.com/wjakob/nanobind.git",
            tag = "v1.9.2",
            build_file = "@//bindings/python:nanobind.BUILD",
            recursive_init_submodules = True,
        )

    if "libpfm" not in native.existing_rules():
        # Downloaded from v4.9.0 tag at https://sourceforge.net/p/perfmon2/libpfm4/ref/master/tags/
        http_archive(
            name = "libpfm",
            build_file = str(Label("//tools:libpfm.BUILD.bazel")),
            sha256 = "5da5f8872bde14b3634c9688d980f68bda28b510268723cc12973eedbab9fecc",
            type = "tar.gz",
            strip_prefix = "libpfm-4.11.0",
            urls = ["https://sourceforge.net/projects/perfmon2/files/libpfm4/libpfm-4.11.0.tar.gz/download"],
        )
