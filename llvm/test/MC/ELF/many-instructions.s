# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null -debug-only=mc-dump

## Test that encodeInstruction may cause a new fragment to be created.
# CHECK: 0 Data Size:16200
# CHECK: 16200 Data Size:180

.rept 16384/10
movabsq $foo, %rax
.endr
