// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+tlb-rmi,+tlbiw,+rme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+tlb-rmi,+tlbiw,+tlbid,+rme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-NO-REGISTER

// Test without using +tlbid - no optional register operand allowed

tlbi vmalle1is, x1
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmalle1is, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmalle1os, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmalls12e1os, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi alle1is, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi alle2is, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi alle3is, x5
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmallws2e1os, x1
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmalls12e1is, x1
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmallws2e1is, x1
// CHECK-ERROR: error: specified tlbi op does not use a register

tlbi vmalle1, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi alle1, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi alle2, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi alle3, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi vmalls12e1, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi paallos, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register

tlbi paall, x1
// CHECK-ERROR: error: specified tlbi op does not use a register
// CHECK-NO-REGISTER: error: specified tlbi op does not use a register
