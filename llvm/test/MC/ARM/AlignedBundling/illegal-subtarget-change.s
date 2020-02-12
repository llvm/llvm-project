# RUN: not --crash llvm-mc -filetype=obj -triple armv7-linux-gnueabi %s -o - 2>&1 | FileCheck %s

        # We cannot switch subtargets mid-bundle
        .syntax unified
        .text
        .bundle_align_mode 4
        .arch armv4t
        bx lr
        .bundle_lock
        bx lr
        .arch armv7a
        movt r0, #0xffff
        movw r0, #0xffff
        .bundle_unlock
        bx lr
# CHECK: LLVM ERROR: A Bundle can only have one Subtarget.
