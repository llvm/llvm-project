// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+poe2 < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+poe2,+tlbid < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-NO-REGISTER

// Test without using +tlbid - no optional register operand allowed

plbi alle2is, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle2os, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle1is, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle1os, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi vmalle1is, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi vmalle1os, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle2isnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle2osnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle1isnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi alle1osnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi vmalle1isnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

plbi vmalle1osnxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register

// Tests where no optional register operand allowed
plbi alle2, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi alle1, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi vmalle1, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi alle2nxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi alle1nxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi vmalle1nxs, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

plbi alle3, x0
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-NO-REGISTER: error: specified plbi op does not use a register

