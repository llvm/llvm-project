import os
import sys
import subprocess

from utils import log


def is_configured_build(build_dir):
    return os.path.exists(os.path.join(build_dir, "build.ninja"))


def configure_llvm_build(build_dir):
    print(f"[patch-coverage] Configuring LLVM build in {build_dir}...")
    subprocess.check_call(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            "llvm",
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_INCLUDE_TESTS=ON",
            "-DLLVM_BUILD_TESTS=ON",
        ]
    )


def ensure_llvm_tools(build_dir):
    print("Making sure we have required tool...")
    if not is_configured_build(build_dir):
        configure_llvm_build(build_dir)

    required_tools = [
        "bin/llvm-lit",
        "bin/FileCheck",
        "bin/count",
        "bin/not",
        "bin/llvm-config",
    ]

    missing = [
        tool
        for tool in required_tools
        if not os.path.exists(os.path.join(build_dir, tool))
    ]

    if not missing:
        print("Yes, We have all the tools for fetching testsuite info.")
        return

    print("Building all required tools.")
    subprocess.check_call(
        [
            "ninja",
            "-C",
            build_dir,
            "FileCheck",
            "count",
            "not",
            "llvm-config",
        ]
    )


def build_llvm(inst_build_dir, binary, allowlist_path):
    try:
        if not is_configured_build(inst_build_dir):
            print("Configuring the instrumented build for patch coverage.")
            subprocess.check_call(
                [
                    "cmake",
                    "-G",
                    "Ninja",
                    "-S",
                    "llvm",
                    "-B",
                    inst_build_dir,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DLLVM_ENABLE_ASSERTIONS=ON",
                    "-DLLVM_INCLUDE_TESTS=ON",
                    "-DLLVM_BUILD_TESTS=ON",
                    "-DLLVM_BUILD_INSTRUMENTED_COVERAGE=ON",
                    "-DLLVM_INDIVIDUAL_TEST_COVERAGE=ON",
                    f"-DCMAKE_C_FLAGS=-fprofile-list={os.path.abspath(allowlist_path)}",
                    f"-DCMAKE_CXX_FLAGS=-fprofile-list={os.path.abspath(allowlist_path)}",
                ]
            )

        target = [
            "FileCheck",
            "count",
            "not",
            "llvm-config",
            "llvm-cov",
            "llvm-profdata",
            "llvm-dwarfdump",
            "llvm-readobj",
            binary,
        ]

        try:
            print("Building the instrumented target")
            subprocess.check_call(["ninja", "-C", inst_build_dir] + target)

        except subprocess.CalledProcessError as ninja_error:
            log(f"Error during Ninja build: {ninja_error}")
            sys.exit(1)

        log("LLVM build completed successfully.\n")

    except subprocess.CalledProcessError as e:
        log("Error during LLVM build:", e)
        sys.exit(1)
