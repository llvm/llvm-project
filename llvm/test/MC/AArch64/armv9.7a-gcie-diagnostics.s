// RUN: not llvm-mc -triple=aarch64 -mattr=+gcie -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// FEAT_GCIE instructions
//------------------------------------------------------------------------------

gsb
// CHECK-ERROR: error: invalid operand for GSB instruction

gicr
// CHECK-ERROR: error: expected register operand

gicr x3, foo
// CHECK-ERROR: error: invalid operand for GICR instruction

gic cdaff
// CHECK-ERROR: error: specified gic op requires a register

gic cdeoi, x3
// CHECK-ERROR: error: specified gic op does not use a register

