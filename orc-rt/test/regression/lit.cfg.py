# -*- Python -*-

import os
import random

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import platform
import mmap

config.name = "ORC-RT"
config.test_format = lit.formats.ShTest()
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.orc_rt_obj_root, "test", "regression")
config.suffixes = [
    ".test"
]

# Regression test-support tools live under test/tools.
test_tools_dir = os.path.join(config.orc_rt_obj_root, "test", "tools")

llvm_config.with_environment(
    "PATH",
    os.path.join(config.orc_rt_obj_root, "tools", "ogre"),
    append_path=True)
llvm_config.with_environment("PATH", test_tools_dir, append_path=True)

llvm_config.use_default_substitutions()

# %{jit} runs JIT'd code under ogre, with llvm-jitlink as the controller. Tests
# that use it must be gated on the llvm-jitlink feature.
ogre = os.path.join(config.orc_rt_obj_root, "tools", "ogre", "ogre")
config.substitutions.append(("%{ogre}", ogre))
llvm_jitlink = llvm_config.use_llvm_tool("llvm-jitlink")
if llvm_jitlink:
    config.available_features.add("llvm-jitlink")
    config.substitutions.append(
        ("%{jit}", "{} -oop-launch={}".format(llvm_jitlink, ogre))
    )


# %{cc} and %{cxx} compile C and C++ for the runtime's target. They default to
# clang and clang++ from the LLVM tools directory. Pass --param orc-rt-cc=<cc>
# or --param orc-rt-cxx=<cxx> to use a different compiler: it must accept
# clang-style options and target the runtime's target without being told to
# (use a wrapper script if necessary). Tests that use %{cc} or %{cxx} must be
# gated on the orc-rt-cc or orc-rt-cxx feature respectively.
def add_compiler(name, clang_name):
    override = lit_config.params.get(name)
    if override:
        compiler = lit.util.which(override)
        if compiler is None:
            lit_config.fatal("{} compiler '{}' not found".format(name, override))
        lit_config.note("using {} override: {}".format(name, compiler))
    else:
        compiler = llvm_config.use_llvm_tool(clang_name)
        if compiler:
            compiler += " --target=" + config.target_triple
    if compiler:
        config.available_features.add(name)
        substitution = "%{" + name[len("orc-rt-") :] + "}"
        config.substitutions.append((substitution, compiler))


add_compiler("orc-rt-cc", "clang")
add_compiler("orc-rt-cxx", "clang++")

# %{mc} assembles its input into an object file for the runtime's target.
# Unlike %{cc}, it can't be overridden, so that object format tests always
# check the runtime against the same assembler. Tests that use it must be
# gated on the llvm-mc feature.
llvm_mc = llvm_config.use_llvm_tool("llvm-mc")
if llvm_mc:
    config.available_features.add("llvm-mc")
    config.substitutions.append(
        (
            "%{mc}",
            "{} -triple={} -filetype=obj".format(llvm_mc, config.target_triple),
        )
    )

# Describe the runtime's target architecture and object format, so that object
# format tests can gate on them:
#   target-arch=<arch>             (arm64 and aarch64 are aliases)
#   target-object-format=<coff|elf|mach-o>
# No object format feature is added for targets not recognized below.
ELF_OS_NAMES = ("linux", "freebsd", "netbsd", "openbsd", "fuchsia", "none", "elf")


def add_target_features():
    arch, _, rest = config.target_triple.partition("-")
    if arch in ("arm64", "aarch64"):
        config.available_features.update(["target-arch=arm64", "target-arch=aarch64"])
    else:
        config.available_features.add("target-arch=" + arch)
    components = rest.split("-")
    if "apple" in components:
        object_format = "mach-o"
    elif any(c.startswith("windows") for c in components):
        object_format = "coff"
    elif any(c.startswith(n) for c in components for n in ELF_OS_NAMES):
        object_format = "elf"
    else:
        return
    config.available_features.add("target-object-format=" + object_format)


add_target_features()


def run_test_tool(name, *args):
    """Run a test-support tool from test/tools and return its stdout.

    Returns None if the tool has not been built yet (so feature probing degrades
    gracefully). Raises if the tool exits with a non-zero status.
    """
    tool = lit.util.which(name, test_tools_dir)
    if tool is None:
        return None
    out, err, exit_code = lit.util.executeCommand([tool, *args])
    if exit_code != 0:
        raise RuntimeError(
            "{} {} failed (exit {}):\n{}".format(name, " ".join(args), exit_code, err)
        )
    return out


# Probe the compiled-in logging configuration from orc-rt-log-check and
# expose it as lit features, so logging tests can gate on the build's backend
# and on which levels are actually emitted:
#   orc-rt-log-backend-<none|printf|os_log>
#   orc-rt-log-level-<error|warning|info|debug>   (one per compiled-in level)
def add_logging_features():
    backend = run_test_tool("orc-rt-log-check", "--print-backend")
    if backend is None:
        return  # tool not built yet; skip feature probing
    config.available_features.add("orc-rt-log-backend-" + backend.strip())
    levels = run_test_tool("orc-rt-log-check", "--print-enabled-levels")
    for level in levels.split():
        config.available_features.add("orc-rt-log-level-" + level.lower())

add_logging_features()

# The os_log delivery tests scrape the unified log (via `log show`), which is
# slow and timing-sensitive, so they are opt-in: pass --param run-os-log-tests=1
# to enable them. They also need the `log` tool. Warn if the tests were
# requested but `log` is unavailable, so the request doesn't silently no-op.
if lit_config.params.get("run-os-log-tests"):
    if lit.util.which("log"):
        config.available_features.add("os-log-show-tests")
        # A per-invocation id (stable across ALLOW_RETRIES) that the delivery
        # test emits and matches, so it can't match a stale record from an
        # earlier run.
        config.substitutions.append(
            ("%{orc-rt-log-uid}", str(random.randint(1, 2**31 - 1)))
        )
    else:
        lit_config.warning(
            "run-os-log-tests was requested, but the 'log' tool was not found; "
            "the os_log delivery tests will be skipped"
        )

# Give logging tests a deterministic baseline: clear any logging environment
# inherited from the developer's shell. Tests opt in with `env ORC_RT_LOG=...`.
for var in ("ORC_RT_LOG", "ORC_RT_LOG_OUTPUT"):
    config.environment.pop(var, None)

if platform.system() == "Darwin":
    config.substitutions.append(("%macos-product-version", platform.mac_ver()[0]))
config.substitutions.append(("%target_triple", config.target_triple))

# The architecture the runtime was built for, so tests can check the triple it
# reports against an independent source.
config.substitutions.append(("%target-arch", config.target_triple.split("-")[0]))

# Add the page size from mmap this allows us to avoid another if statement as
# it would likely need ctypes for windows as it does not support sysconf
config.substitutions.append(("%host-page-size", str(mmap.PAGESIZE)))

# Add host OS and arch substitutions for host-detection tests.
config.substitutions.append(("%host-arch", platform.machine()))
if platform.system() == "Darwin":
    config.substitutions.append(("%host-os", "macosx"))
else:
    config.substitutions.append(("%host-os", platform.system().lower()))
