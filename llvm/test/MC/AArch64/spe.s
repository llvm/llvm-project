// RUN: llvm-mc -triple aarch64 -mattr +spe-eef -show-encoding %s 2>%t | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -mattr +v8.7a %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SPE-EEF-ERR %s
// RUN: not llvm-mc -triple aarch64 < %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SPE-EEF-ERR %s

msr PMSNEVFR_EL1, x0
mrs x1, PMSNEVFR_EL1
// CHECK: msr     PMSNEVFR_EL1, x0        // encoding: [0x20,0x99,0x18,0xd5]
// CHECK: mrs     x1, PMSNEVFR_EL1        // encoding: [0x21,0x99,0x38,0xd5]

// CHECK-NO-SPE-EEF-ERR: [[@LINE-5]]:5: error: expected writable system register or pstate
// CHECK-NO-SPE-EEF-ERR: [[@LINE-5]]:9: error: expected readable system register
