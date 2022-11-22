// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+mec < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2>&1 | FileCheck --check-prefix=CHECK-NO-MEC %s

          mrs x0, MECIDR_EL2
// CHECK: mrs   x0, MECIDR_EL2       // encoding: [0xe0,0xa8,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, MECID_P0_EL2
// CHECK: mrs   x0, MECID_P0_EL2      // encoding: [0x00,0xa8,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, MECID_A0_EL2
// CHECK: mrs   x0, MECID_A0_EL2      // encoding: [0x20,0xa8,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, MECID_P1_EL2
// CHECK: mrs   x0, MECID_P1_EL2      // encoding: [0x40,0xa8,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, MECID_A1_EL2
// CHECK: mrs   x0, MECID_A1_EL2      // encoding: [0x60,0xa8,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, VMECID_P_EL2
// CHECK: mrs   x0, VMECID_P_EL2     // encoding: [0x00,0xa9,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, VMECID_A_EL2
// CHECK: mrs   x0, VMECID_A_EL2     // encoding: [0x20,0xa9,0x3c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          mrs x0, MECID_RL_A_EL3
// CHECK: mrs   x0, MECID_RL_A_EL3   // encoding: [0x20,0xaa,0x3e,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:19: error: expected readable system register
          msr MECID_P0_EL2,    x0
// CHECK: msr   MECID_P0_EL2, x0      // encoding: [0x00,0xa8,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr MECID_A0_EL2,    x0
// CHECK: msr   MECID_A0_EL2, x0      // encoding: [0x20,0xa8,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr MECID_P1_EL2,    x0
// CHECK: msr   MECID_P1_EL2, x0      // encoding: [0x40,0xa8,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr MECID_A1_EL2,    x0
// CHECK: msr   MECID_A1_EL2, x0      // encoding: [0x60,0xa8,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr VMECID_P_EL2,   x0
// CHECK: msr   VMECID_P_EL2, x0     // encoding: [0x00,0xa9,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr VMECID_A_EL2,   x0
// CHECK: msr   VMECID_A_EL2, x0     // encoding: [0x20,0xa9,0x1c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate
          msr MECID_RL_A_EL3, x0
// CHECK: msr   MECID_RL_A_EL3, x0   // encoding: [0x20,0xaa,0x1e,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:15: error: expected writable system register or pstate

          dc cigdpae, x0
// CHECK: dc cigdpae, x0             // encoding: [0x00,0x7e,0x0c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:14: error: DC CIGDPAE requires: mec
          dc cipae, x0
// CHECK: dc cipae, x0               // encoding: [0xe0,0x7e,0x0c,0xd5]
// CHECK-NO-MEC: [[@LINE-2]]:14: error: DC CIPAE requires: mec
