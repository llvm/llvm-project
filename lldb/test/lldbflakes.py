"""Rerun policy shared by the API and Shell test formats."""

import re

import lit.Test

# Failures that come from flaky infrastructure rather than from the test
# itself. Rerunning masks real bugs, so this is deliberately not a blanket
# retry of every failure. Adding an entry is a last resort, reserved for a
# defect outside LLDB's control that cannot be worked around any other way.
KNOWN_FLAKES = [
    # macOS 26.3 denies debugserver permission to attach when too many debug
    # sessions start at once, which surfaces as an immediate process exit.
    re.compile(r"process exited with status -1"),
]

MAX_ATTEMPTS = 3


def _hit_known_flake(output):
    return any(flake.search(output) for flake in KNOWN_FLAKES)


def execute_with_reruns(execute_once):
    """Run execute_once, which returns a lit.Test.Result, until it stops
    failing with a known flake."""
    for attempt in range(MAX_ATTEMPTS):
        result = execute_once()
        if result.code != lit.Test.FAIL or not _hit_known_flake(result.output):
            break

    # A pass that needed a rerun is reported apart from a clean pass so that
    # the flakiness stays visible.
    if attempt > 0:
        if result.code == lit.Test.PASS:
            result.code = lit.Test.FLAKYPASS
        result.attempts = attempt + 1
        result.max_allowed_attempts = MAX_ATTEMPTS

    return result
