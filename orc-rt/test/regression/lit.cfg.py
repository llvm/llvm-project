# -*- Python -*-

import os
import random

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

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
