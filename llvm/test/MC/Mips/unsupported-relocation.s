# RUN: not llvm-mc -triple mips-unknown-linux -filetype=obj -o %t %s 2>&1 \
# RUN:  |  FileCheck %s

# Check that we emit an error for unsupported relocations instead of crashing.

        .globl x

        .data
foo:
        .byte   x
# CHECK: :[[@LINE-1]]:17: error: MIPS does not support one byte relocations
        .byte   x+1
# CHECK: :[[@LINE-1]]:17: error: MIPS does not support one byte relocations
        .quad   x-foo
# CHECK: :[[@LINE-1]]:17: error: MIPS does not support 64-bit PC-relative relocations
