# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helper macros to configure the LLVM overlay project."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(":zlib.bzl", "llvm_zlib_disable", "llvm_zlib_system")
load(":terminfo.bzl", "llvm_terminfo_disable", "llvm_terminfo_system")

# Directory of overlay files relative to WORKSPACE
DEFAULT_OVERLAY_PATH = "llvm-project-overlay"

DEFAULT_TARGETS = [
    "AArch64",
    "AMDGPU",
    "ARM",
    "AVR",
    "BPF",
    "Hexagon",
    "Lanai",
    "Mips",
    "MSP430",
    "NVPTX",
    "PowerPC",
    "RISCV",
    "Sparc",
    "SystemZ",
    "WebAssembly",
    "X86",
    "XCore",
]

def _overlay_directories(repository_ctx):
    src_path = repository_ctx.path(Label("//:WORKSPACE")).dirname
    bazel_path = src_path.get_child("utils").get_child("bazel")
    overlay_path = bazel_path.get_child("llvm-project-overlay")
    script_path = bazel_path.get_child("overlay_directories.py")

    python_bin = repository_ctx.which("python3")
    if not python_bin:
        # Windows typically just defines "python" as python3. The script itself
        # contains a check to ensure python3.
        python_bin = repository_ctx.which("python")

    if not python_bin:
        fail("Failed to find python3 binary")

    cmd = [
        python_bin,
        script_path,
        "--src",
        src_path,
        "--overlay",
        overlay_path,
        "--target",
        ".",
    ]
    exec_result = repository_ctx.execute(cmd, timeout = 20)

    if exec_result.return_code != 0:
        fail(("Failed to execute overlay script: '{cmd}'\n" +
              "Exited with code {return_code}\n" +
              "stdout:\n{stdout}\n" +
              "stderr:\n{stderr}\n").format(
            cmd = " ".join([str(arg) for arg in cmd]),
            return_code = exec_result.return_code,
            stdout = exec_result.stdout,
            stderr = exec_result.stderr,
        ))

def _llvm_configure_impl(repository_ctx):
    _overlay_directories(repository_ctx)

    # Create a starlark file with the requested LLVM targets.
    targets = repository_ctx.attr.targets
    repository_ctx.file(
        "llvm/targets.bzl",
        content = "llvm_targets = " + str(targets),
        executable = False,
    )

llvm_configure = repository_rule(
    implementation = _llvm_configure_impl,
    local = True,
    configure = True,
    attrs = {
        "targets": attr.string_list(default = DEFAULT_TARGETS),
    },
)

def llvm_disable_optional_support_deps():
    maybe(
        llvm_zlib_disable,
        name = "llvm_zlib",
    )

    maybe(
        llvm_terminfo_disable,
        name = "llvm_terminfo",
    )

def llvm_use_system_support_deps():
    maybe(
        llvm_zlib_system,
        name = "llvm_zlib",
    )

    maybe(
        llvm_terminfo_system,
        name = "llvm_terminfo",
    )
