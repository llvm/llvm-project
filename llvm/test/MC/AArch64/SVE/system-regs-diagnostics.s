// RUN: not llvm-mc -triple aarch64 -mattr=+sve -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SVE
// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOSVE


// --------------------------------------------------------------------------//
// ID_AA64ZFR0_EL1 is read-only

msr ID_AA64ZFR0_EL1, x3
// CHECK-SVE: error: expected writable system register or pstate
// CHECK-SVE-NEXT:         msr ID_AA64ZFR0_EL1, x3


// --------------------------------------------------------------------------//
// Check that the other SVE registers are only readable/writable when
// the +sve attribute is set.

mrs x3, ID_AA64ZFR0_EL1
// CHECK-NOSVE: error: expected readable system register
// CHECK-NOSVE: mrs x3, ID_AA64ZFR0_EL1

mrs x3, ZCR_EL1
// CHECK-NOSVE: error: expected readable system register
// CHECK-NOSVE-NEXT: mrs x3, ZCR_EL1

mrs x3, ZCR_EL2
// CHECK-NOSVE: error: expected readable system register
// CHECK-NOSVE-NEXT: mrs x3, ZCR_EL2

mrs x3, ZCR_EL3
// CHECK-NOSVE: error: expected readable system register
// CHECK-NOSVE-NEXT: mrs x3, ZCR_EL3

mrs x3, ZCR_EL12
// CHECK-NOSVE: error: expected readable system register
// CHECK-NOSVE-NEXT: mrs x3, ZCR_EL12

msr ZCR_EL1, x3
// CHECK-NOSVE: error: expected writable system register or pstate
// CHECK-NOSVE-NEXT: msr ZCR_EL1, x3

msr ZCR_EL2, x3
// CHECK-NOSVE: error: expected writable system register or pstate
// CHECK-NOSVE-NEXT: msr ZCR_EL2, x3

msr ZCR_EL3, x3
// CHECK-NOSVE: error: expected writable system register or pstate
// CHECK-NOSVE-NEXT: msr ZCR_EL3, x3

msr ZCR_EL12, x3
// CHECK-NOSVE: error: expected writable system register or pstate
// CHECK-NOSVE-NEXT: msr ZCR_EL12, x3
