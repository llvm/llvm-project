# RUN: not llvm-mc -triple mips-unknown-linux -filetype=obj %s 2>%t
# RUN: FileCheck %s < %t

# Check that we emit an error for unsupported relocations instead of crashing.

        .globl x

        .data
foo:
        .byte   x
        .byte   x+1

# CHECK: LLVM ERROR: MIPS does not support one byte relocations
