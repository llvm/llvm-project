// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2>&1 | FileCheck %s

msr PM, #2
// CHECK:         error: immediate must be an integer in range [0, 1].
// CHECK-NEXT:    msr PM, #2
// CHECK-NEXT:    ^
