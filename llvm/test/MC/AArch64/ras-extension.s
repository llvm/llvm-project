// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+ras < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mcpu=cortex-a55 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mcpu=cortex-a75 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mcpu=tsv110 < %s | FileCheck %s

  esb
// CHECK: esb                             // encoding: [0x1f,0x22,0x03,0xd5]

  msr errselr_el1, x0
  msr errselr_el1, x15
  msr errselr_el1, x25
  msr erxctlr_el1, x1
  msr erxstatus_el1, x2
  msr erxaddr_el1, x3
  msr erxmisc0_el1, x4
  msr erxmisc1_el1, x5
  msr disr_el1, x6
  msr vdisr_el2, x7
  msr vsesr_el2, x8
// CHECK: msr     ERRSELR_EL1, x0         // encoding: [0x20,0x53,0x18,0xd5]
// CHECK: msr     ERRSELR_EL1, x15        // encoding: [0x2f,0x53,0x18,0xd5]
// CHECK: msr     ERRSELR_EL1, x25        // encoding: [0x39,0x53,0x18,0xd5]
// CHECK: msr     ERXCTLR_EL1, x1         // encoding: [0x21,0x54,0x18,0xd5]
// CHECK: msr     ERXSTATUS_EL1, x2       // encoding: [0x42,0x54,0x18,0xd5]
// CHECK: msr     ERXADDR_EL1, x3         // encoding: [0x63,0x54,0x18,0xd5]
// CHECK: msr     ERXMISC0_EL1, x4        // encoding: [0x04,0x55,0x18,0xd5]
// CHECK: msr     ERXMISC1_EL1, x5        // encoding: [0x25,0x55,0x18,0xd5]
// CHECK: msr     DISR_EL1, x6            // encoding: [0x26,0xc1,0x18,0xd5]
// CHECK: msr     VDISR_EL2, x7           // encoding: [0x27,0xc1,0x1c,0xd5]
// CHECK: msr     VSESR_EL2, x8           // encoding: [0x68,0x52,0x1c,0xd5]

  mrs x0, errselr_el1
  mrs x15, errselr_el1
  mrs x25, errselr_el1
  mrs x1, erxctlr_el1
  mrs x2, erxstatus_el1
  mrs x3, erxaddr_el1
  mrs x4, erxmisc0_el1
  mrs x5, erxmisc1_el1
  mrs x6, disr_el1
  mrs x7, vdisr_el2
  mrs x8, vsesr_el2
// CHECK: mrs     x0, ERRSELR_EL1         // encoding: [0x20,0x53,0x38,0xd5]
// CHECK: mrs     x15, ERRSELR_EL1        // encoding: [0x2f,0x53,0x38,0xd5]
// CHECK: mrs     x25, ERRSELR_EL1        // encoding: [0x39,0x53,0x38,0xd5]
// CHECK: mrs     x1, ERXCTLR_EL1         // encoding: [0x21,0x54,0x38,0xd5]
// CHECK: mrs     x2, ERXSTATUS_EL1       // encoding: [0x42,0x54,0x38,0xd5]
// CHECK: mrs     x3, ERXADDR_EL1         // encoding: [0x63,0x54,0x38,0xd5]
// CHECK: mrs     x4, ERXMISC0_EL1        // encoding: [0x04,0x55,0x38,0xd5]
// CHECK: mrs     x5, ERXMISC1_EL1        // encoding: [0x25,0x55,0x38,0xd5]
// CHECK: mrs     x6, DISR_EL1            // encoding: [0x26,0xc1,0x38,0xd5]
// CHECK: mrs     x7, VDISR_EL2           // encoding: [0x27,0xc1,0x3c,0xd5]
// CHECK: mrs     x8, VSESR_EL2           // encoding: [0x68,0x52,0x3c,0xd5]

  mrs x0, erridr_el1
  mrs x1, erxfr_el1
// CHECK: mrs     x0, ERRIDR_EL1          // encoding: [0x00,0x53,0x38,0xd5]
// CHECK: mrs     x1, ERXFR_EL1           // encoding: [0x01,0x54,0x38,0xd5]
