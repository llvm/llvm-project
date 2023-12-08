// RUN: llvm-mc -triple aarch64 -mattr +spe-eef -show-encoding %s 2>%t | FileCheck %s

msr PMSNEVFR_EL1, x0
mrs x1, PMSNEVFR_EL1
// CHECK: msr     PMSNEVFR_EL1, x0        // encoding: [0x20,0x99,0x18,0xd5]
// CHECK: mrs     x1, PMSNEVFR_EL1        // encoding: [0x21,0x99,0x38,0xd5]
