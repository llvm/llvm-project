# Introduction

*Warning* The Bazel build is experimental and best-effort, supported in line
with the policy for
[LLVM's peripheral support tier](https://llvm.org/docs/SupportPolicy.html).
LLVM's official build system is CMake. If in doubt use that. If you make changes
to LLVM, you're expected to update the CMake build but you don't need to update
Bazel build files. Reviewers should not ask authors to update Bazel build files
unless the author has opted in to support Bazel. Keeping the Bazel build files
up-to-date is on the people who use the Bazel build.

[Bazel](https://bazel.build/) is a multi-language build system focused on
reproducible builds to enable dependency analysis and caching for fast
incremental builds.

The main motivation behind the existence of an LLVM Bazel build is that a number
of projects that depend on LLVM use Bazel, and Bazel works best when it knows
about the whole source tree (as opposed to installing artifacts coming from
another build system). Community members are also welcome to use Bazel for their
own development as long as they continue to maintain the official CMake build
system. See also, the
[proposal](https://github.com/llvm/llvm-www/blob/main/proposals/LP0002-BazelBuildConfiguration.md)
for adding this configuration.

# Quick Start

1. `git clone https://github.com/llvm/llvm-project.git; cd llvm-project` if
   you don't have a checkout yet.
2. Install Bazel at the version indicated by [.bazelversion](./bazelversion),
   following the official instructions, if you don't have it installed yet:
   https://docs.bazel.build/versions/master/install.html.
3. `cd utils/bazel`
4. `bazel build --config=generic_clang @llvm-project//...` (if building on Unix
   with Clang). `--config=generic_gcc` and `--config=msvc` are also available.


# Configuration

The repository `.bazelrc` will import user-specific settings from a
`user.bazelrc` file (in addition to the standard locations). Adding your typical
config setting is recommended.

```.bazelrc
build --config=generic_clang
```

You can enable
[disk caching](https://docs.bazel.build/versions/master/remote-caching.html#disk-cache),
which will cache build results

```.bazelrc
build --disk_cache=~/.cache/bazel-disk-cache
```

You can instruct Bazel to use a ramdisk for its sandboxing operations via
[--sandbox_base](https://docs.bazel.build/versions/master/command-line-reference.html#flag--sandbox_base),
which can help avoid IO bottlenecks for the symlink stragegy used for
sandboxing. This is especially important with many inputs and many cores (see
https://github.com/bazelbuild/bazel/issues/11868):

```.bazelrc
build --sandbox_base=/dev/shm
```

Bear in mind that this requires that your ramdisk is of sufficient size to hold
any temporary files. Anecdotally, 1GB should be sufficient.

# Coverage

The LLVM, MLIR, and Clang subprojects have configurations for Linux (Clang and
GCC), Mac (Clang and GCC), and Windows (MSVC). Configuration options that are
platform-specific are selected for in defines. Many are also hardcoded to the
values currently used by all supported configurations. If there is a
configuration you'd like to use that isn't supported, please send a patch.

# Usage

To use in dependent projects using Bazel, you can import LLVM and then use the
provided configuration rule. See example usage in the `examples/` directory.
