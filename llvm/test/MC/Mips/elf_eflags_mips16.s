# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o -| llvm-readobj -h - | FileCheck %s

# This *MUST* match the output of 'gcc -c' compiled with the same triple.
# CHECK: Flags [ (0x54001004)

        .set mips16
f:
        nop
