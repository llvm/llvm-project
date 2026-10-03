import os
import shutil
import sys
import subprocess

from utils import log
from utils import resolve_projects


def is_configured_build(build_dir):
    return os.path.exists(os.path.join(build_dir, "build.ninja"))


def configure_llvm_build(build_dir, projects):
    print(f"[patch-coverage] Configuring LLVM build in {build_dir}...")

    cmake_cmd = [
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

    if projects:
        cmake_cmd.append(f"-DLLVM_ENABLE_PROJECTS={projects}")

    if projects and "lldb" in projects:
        cmake_cmd.append(f"-DLLVM_ENABLE_RUNTIMES=libcxx")
        cmake_cmd.append(f"-DPYTHON_EXECUTABLE={shutil.which('python3')}")

    print("CMake cmd:", " ".join(cmake_cmd))
    subprocess.check_call(cmake_cmd)


def ensure_llvm_tools(build_dir, projects, binary):
    print("Making sure we have required tools...")

    if not is_configured_build(build_dir):
        configure_llvm_build(build_dir, projects)

    required_tools = [
        "bin/llvm-lit",
        "bin/FileCheck",
        "bin/count",
        "bin/not",
        "bin/llvm-config",
    ]

    missing_tools = [
        tool
        for tool in required_tools
        if not os.path.exists(os.path.join(build_dir, tool))
    ]

    if missing_tools:
        print("Building missing core tools:", missing_tools)
        subprocess.check_call(
            ["ninja", "-C", build_dir, "FileCheck", "count", "not", "llvm-config"]
        )
    else:
        print("Core tools already present.")

    # Handle binary-specific requirements
    extra_targets = []

    if binary == "clang-tidy":
        clang_path = os.path.join(build_dir, "bin", "clang")
        if not os.path.exists(clang_path):
            extra_targets.append("clang")

    if binary == "lldb":
        lldb_path = os.path.join(build_dir, "bin", "lldb")
        if not os.path.exists(lldb_path):
            extra_targets.append("lldb")

    if extra_targets:
        print("Building required targets to parse testsuite info:", extra_targets)
        subprocess.check_call(["ninja", "-C", build_dir] + extra_targets)
    else:
        print("All targets already present to get testsuite info.")


def build_llvm(inst_build_dir, build_dir, binary, projects, allowlist_path):
    try:
        if not is_configured_build(inst_build_dir):
            print("Configuring the instrumented build for patch coverage.")

            cmake_cmd = [
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

            if projects:
                cmake_cmd.append(f"-DLLVM_ENABLE_PROJECTS={projects}")

            if projects and "lldb" in projects:
                cmake_cmd.append(f"-DLLVM_ENABLE_RUNTIMES=libcxx")
                cmake_cmd.append(f"-DPYTHON_EXECUTABLE={shutil.which('python3')}")

            print("CMake cmd:", " ".join(cmake_cmd))
            subprocess.check_call(cmake_cmd)

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

        if binary == "clang-tidy":
            target.append("clang")

        if binary == "lldb":
            target.extend(["clang", "lldb", "dsymutil"])

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
