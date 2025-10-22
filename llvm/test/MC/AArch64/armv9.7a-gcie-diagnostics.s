// RUN: not llvm-mc -triple=aarch64 -mattr=+gcie -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-REQUIRES-GCIE

//------------------------------------------------------------------------------
// FEAT_GCIE instructions
//------------------------------------------------------------------------------

gsb
// CHECK-ERROR: error: invalid operand for GSB instruction

gsb ack
// CHECK-REQUIRES-GCIE: GSB ack requires: gcie

gicr
// CHECK-ERROR: error: expected register operand

gicr x3, foo
// CHECK-ERROR: error: invalid operand for GICR instruction

gicr x3, cdnmia
// CHECK-REQUIRES-GCIE: GICR cdnmia requires: gcie

gic cdaff
// CHECK-ERROR: error: specified gic op requires a register

gic cdaff, x3
// CHECK-REQUIRES-GCIE: GIC cdaff requires: gcie
