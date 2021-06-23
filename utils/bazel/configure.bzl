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
    "BPF",
    "Hexagon",
    "Lanai",
    "NVPTX",
    "PowerPC",
    "RISCV",
    "Sparc",
    "WebAssembly",
    "X86",
]

def _is_absolute(path):
    """Returns `True` if `path` is an absolute path.

    Args:
      path: A path (which is a string).
    Returns:
      `True` if `path` is an absolute path.
    """
    return path.startswith("/") or (len(path) > 2 and path[1] == ":")

def _join_path(a, b):
    if _is_absolute(b):
        return b
    return str(a) + "/" + str(b)

def _overlay_directories(repository_ctx):
    src_workspace_path = repository_ctx.path(
        repository_ctx.attr.src_workspace,
    ).dirname

    src_path = _join_path(src_workspace_path, repository_ctx.attr.src_path)

    overlay_workspace_path = repository_ctx.path(
        repository_ctx.attr.overlay_workspace,
    ).dirname
    overlay_path = _join_path(
        overlay_workspace_path,
        repository_ctx.attr.overlay_path,
    )

    overlay_script = repository_ctx.path(
        repository_ctx.attr._overlay_script,
    )
    python_bin = repository_ctx.which("python3")
    if not python_bin:
        # Windows typically just defines "python" as python3. The script itself
        # contains a check to ensure python3.
        python_bin = repository_ctx.which("python")

    if not python_bin:
        fail("Failed to find python3 binary")

    cmd = [
        python_bin,
        overlay_script,
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
        "_overlay_script": attr.label(
            default = Label("//:overlay_directories.py"),
            allow_single_file = True,
        ),
        "overlay_workspace": attr.label(default = Label("//:WORKSPACE")),
        "overlay_path": attr.string(default = DEFAULT_OVERLAY_PATH),
        "src_workspace": attr.label(default = Label("//:WORKSPACE")),
        "src_path": attr.string(mandatory = True),
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
