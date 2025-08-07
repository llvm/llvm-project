## Checks the size of an internal MC structure that is different on 32-bit.
# REQUIRES: asserts, llvm-64-bits
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null -debug-only=mc-dump 2>&1 | grep -E -o '[0-9]+ Data Size:[0-9]+' | FileCheck %s

## Test that encodeInstruction may cause a new fragment to be created.
# CHECK: 0 Data Size:16220
# CHECK: 16220 Data Size:160

.rept 16384/10
movabsq $foo, %rax
.endr
