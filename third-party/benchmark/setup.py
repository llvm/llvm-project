import contextlib
import os
import platform
import re
import shutil
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import setuptools
from setuptools.command import build_ext

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# hardcoded SABI-related options. Requires that each Python interpreter
# (hermetic or not) participating is of the same major-minor version.
py_limited_api = sys.version_info >= (3, 12)
options = {"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {}


def is_cibuildwheel() -> bool:
    return os.getenv("CIBUILDWHEEL") is not None


@contextlib.contextmanager
def _maybe_patch_toolchains() -> Generator[None, None, None]:
    """
    Patch rules_python toolchains to ignore root user error
    when run in a Docker container on Linux in cibuildwheel.
    """

    def fmt_toolchain_args(matchobj):
        suffix = "ignore_root_user_error = True"
        callargs = matchobj.group(1)
        # toolchain def is broken over multiple lines
        if callargs.endswith("\n"):
            callargs = callargs + "    " + suffix + ",\n"
        # toolchain def is on one line.
        else:
            callargs = callargs + ", " + suffix
        return "python.toolchain(" + callargs + ")"

    CIBW_LINUX = is_cibuildwheel() and IS_LINUX
    module_bazel = Path("MODULE.bazel")
    content: str = module_bazel.read_text()
    try:
        if CIBW_LINUX:
            module_bazel.write_text(
                re.sub(
                    r"python.toolchain\(([\w\"\s,.=]*)\)",
                    fmt_toolchain_args,
                    content,
                )
            )
        yield
    finally:
        if CIBW_LINUX:
            module_bazel.write_text(content)


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, name: str, bazel_target: str, **kwargs: Any):
        super().__init__(name=name, sources=[], **kwargs)

        self.bazel_target = bazel_target
        stripped_target = bazel_target.split("//")[-1]
        self.relpath, self.target_name = stripped_target.split(":")


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        # explicitly call `bazel shutdown` for graceful exit
        self.spawn(["bazel", "shutdown"])

    def copy_extensions_to_source(self):
        """
        Copy generated extensions into the source tree.
        This is done in the ``bazel_build`` method, so it's not necessary to
        do again in the `build_ext` base class.
        """

    def bazel_build(self, ext: BazelExtension) -> None:  # noqa: C901
        """Runs the bazel build to create the package."""
        temp_path = Path(self.build_temp)

        # We round to the minor version, which makes rules_python
        # look up the latest available patch version internally.
        python_version = "{}.{}".format(*sys.version_info[:2])

        bazel_argv = [
            "bazel",
            "run",
            ext.bazel_target,
            f"--symlink_prefix={temp_path / 'bazel-'}",
            f"--compilation_mode={'dbg' if self.debug else 'opt'}",
            # C++17 is required by nanobind
            f"--cxxopt={'/std:c++17' if IS_WINDOWS else '-std=c++17'}",
            f"--@rules_python//python/config_settings:python_version={python_version}",
        ]

        if ext.py_limited_api:
            bazel_argv += ["--@nanobind_bazel//:py-limited-api=cp312"]

        if IS_WINDOWS:
            # Link with python*.lib.
            for library_dir in self.library_dirs:
                bazel_argv.append("--linkopt=/LIBPATH:" + library_dir)
        elif IS_MAC:
            # C++17 needs macOS 10.14 at minimum
            bazel_argv.append("--macos_minimum_os=10.14")

        with _maybe_patch_toolchains():
            self.spawn(bazel_argv)

        if IS_WINDOWS:
            suffix = ".pyd"
        else:
            suffix = ".abi3.so" if ext.py_limited_api else ".so"

        # copy the Bazel build artifacts into setuptools' libdir,
        # from where the wheel is built.
        pkgname = "google_benchmark"
        pythonroot = Path("bindings") / "python" / "google_benchmark"
        srcdir = temp_path / "bazel-bin" / pythonroot
        if not self.inplace:
            libdir = Path(self.build_lib) / pkgname
        else:
            build_py = self.get_finalized_command("build_py")
            libdir = Path(build_py.get_package_dir(pkgname))

        for root, dirs, files in os.walk(srcdir, topdown=True):
            # exclude runfiles directories and children.
            dirs[:] = [d for d in dirs if "runfiles" not in d]

            for f in files:
                fp = Path(f)
                should_copy = False
                # we do not want the bare .so file included
                # when building for ABI3, so we require a
                # full and exact match on the file extension.
                if "".join(fp.suffixes) == suffix or fp.suffix == ".pyi":
                    should_copy = True
                elif Path(root) == srcdir and f == "py.typed":
                    # copy py.typed, but only at the package root.
                    should_copy = True

                if should_copy:
                    shutil.copyfile(root / fp, libdir / fp)


setuptools.setup(
    cmdclass={"build_ext": BuildBazelExtension},
    package_data={"google_benchmark": ["py.typed", "*.pyi"]},
    ext_modules=[
        BazelExtension(
            name="google_benchmark._benchmark",
            bazel_target="//bindings/python/google_benchmark:benchmark_stubgen",
            py_limited_api=py_limited_api,
        )
    ],
    options=options,
)
