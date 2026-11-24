// RUN: llvm-mc -triple bpfel --mcpu=v5 --assemble --filetype=obj %s \
// RUN:   | llvm-objdump -d --mattr=+alu32 - \
// RUN:   | FileCheck %s

// CHECK: 2f 21 01 00 00 00 00 00     r1 uh*= r2
// CHECK: 2f 43 02 00 00 00 00 00     r3 sh*= r4
// CHECK: 27 05 01 00 fe ff ff ff     r5 uh*= -0x2
// CHECK: 27 06 02 00 fd ff ff ff     r6 sh*= -0x3
r1 uh*= r2
r3 sh*= r4
r5 uh*= -2
r6 sh*= -3
