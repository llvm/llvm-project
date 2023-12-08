# RUN: not llvm-mca -march=arm -mcpu=swift -all-views=false 2>&1 < %s | FileCheck %s
# D54648 results in this test to become valid.
# XFAIL: *

add r3, r1, r12, lsl #2

# CHECK:      error: unable to resolve scheduling class for write variant.
# CHECK-NEXT: note: instruction:    add r3, r1, r12, lsl #2
