#!/usr/bin/env python3
"""Builds a Win64 unwinding test: compiles the harness and links it with the
object file produced by llc from the test.

usage: build-win64-unwind-harness.py <harness.c> <test.obj> <out.exe>

The compiler is the one LLVM was built with (LLVM_TEST_HOST_CC, set by
lit.local.cfg): cl.exe, clang-cl.exe or clang.exe targeting Windows. It may be
given with arguments, for example "clang.exe --driver-mode=cl".
"""

import os
import shlex
import subprocess
import sys


def main():
    harness, obj, exe = sys.argv[1:4]
    cc = os.environ.get("LLVM_TEST_HOST_CC")
    if not cc:
        sys.exit("LLVM_TEST_HOST_CC is not set")

    words = [w.strip('"') for w in shlex.split(cc, posix=False)]
    name = os.path.basename(words[0]).lower()
    cl_style = (
        name in ("cl", "cl.exe", "clang-cl", "clang-cl.exe")
        or "--driver-mode=cl" in words
    )

    # The linker must not link incrementally: that makes function addresses the
    # addresses of jump thunks, which the harness compares with the addresses
    # the unwinder reports. link.exe and lld-link both read _LINK_, whichever
    # driver runs them.
    env = dict(os.environ)
    env["_LINK_"] = (env.get("_LINK_", "") + " /INCREMENTAL:NO").strip()

    # The harness single-steps, so exceptions must be handled wherever they
    # happen in the __try, not only in calls (/EHa, -fasync-exceptions).
    if cl_style:
        # Compile and link separately: clang-cl rejects /Fo with a file name
        # when the command line also has an object file.
        harness_obj = os.path.splitext(exe)[0] + ".harness.obj"
        cmds = [
            words + ["/nologo", "/O1", "/EHa", "/c", "/Fo" + harness_obj, harness],
            words + ["/nologo", harness_obj, obj, "/Fe" + exe],
        ]
    else:
        cmds = [words + ["-O1", "-fasync-exceptions", "-o", exe, harness, obj]]

    for cmd in cmds:
        print("+ " + " ".join(cmd))
        sys.stdout.flush()
        status = subprocess.call(cmd, env=env)
        if status:
            sys.exit(status)


if __name__ == "__main__":
    main()
