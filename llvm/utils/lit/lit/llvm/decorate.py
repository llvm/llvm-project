"""`--param decorate=PROG[@N]` prepends PROG to pipeline stage N (0-indexed,
default 0) of every RUN line, so the downstream tool can be run under a
debugger, profiler, etc. Examples:

    --param 'decorate=gdb --args'   # run stage 0 under gdb
    --param decorate=perf            # profile stage 0
    --param decorate=time@1          # time the second stage

A stage index past the end of a pipeline is a no-op for that RUN line."""

import re

from lit.llvm.fn_param import add_capture_sub

_TRAILING_INDEX = re.compile(r"^(.+)@(\d+)$")


def install(config, lit_config):
    prog = lit_config.params.get("decorate")
    if not prog:
        return
    m = _TRAILING_INDEX.match(prog)
    if m:
        prog, stage = m.group(1), m.group(2)
    else:
        stage = "0"
    # Match (optional %dbg marker)(stage many `|`-delimited stages and their
    # bars) and insert PROG after, preserving everything captured.
    pattern = r"^(%dbg\([^)]*\)\s*)?((?:[^|]*\|\s*){" + stage + r"})"
    add_capture_sub(config, pattern, r"\1\2" + prog + " ")
