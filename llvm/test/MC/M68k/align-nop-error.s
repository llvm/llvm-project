# RUN: not --crash llvm-mc -triple=m68k -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

        .text
        .asciz "ab"
        .p2align 1

# CHECK: LLVM ERROR: unable to write nop sequence of 1 bytes
