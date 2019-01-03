// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+ssbs  < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-ssbs  < %s 2>&1 | FileCheck %s --check-prefix=NOSPECID

msr SSBS, #2

// CHECK:         error: immediate must be an integer in range [0, 1].
// CHECK-NEXT:    msr SSBS, #2
// NOSPECID:      error: expected writable system register or pstate
// NOSPECID-NEXT: msr {{ssbs|SSBS}}, #2
