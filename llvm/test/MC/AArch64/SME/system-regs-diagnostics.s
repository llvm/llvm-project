// RUN: not llvm-mc -triple aarch64 -mattr=+sme -show-encoding < %s 2>&1 | FileCheck %s


// --------------------------------------------------------------------------//
// Check read-only

msr ID_AA64SMFR0_EL1, x3
// CHECK: error: expected writable system register or pstate
// CHECK-NEXT: msr ID_AA64SMFR0_EL1, x3

msr SMIDR_EL1, x3
// CHECK: error: expected writable system register or pstate
// CHECK-NEXT: msr SMIDR_EL1, x3

// --------------------------------------------------------------------------//
// Check MSR SVCR immediate is in range [0, 1]

msr SVCRSM, #-1
// CHECK: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: msr SVCRSM, #-1

msr SVCRZA, #2
// CHECK: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: msr SVCRZA, #2

msr SVCRSMZA, #4
// CHECK: error: immediate must be an integer in range [0, 1].
// CHECK-NEXT: msr SVCRSMZA, #4
