// +the required for RCWSMASK_EL1, RCWMASK_EL1
// +el2vmsa required for TTBR0_EL2 (VSCTLR_EL2), VTTBR_EL2
// +vh required for TTBR1_EL2

// RUN: not llvm-mc -triple aarch64 -mattr=+d128,+the,+el2vmsa,+vh -show-encoding %s -o - 2> %t | FileCheck %s
// RUN: FileCheck %s --input-file=%t --check-prefix=ERRORS

// RUN: not llvm-mc -triple aarch64 -mattr=+the,+el2vmsa,+vh -show-encoding %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR-NO-D128

          msrr  TTBR0_EL1, x0, x1
// CHECK: msrr  TTBR0_EL1, x0, x1           // encoding: [0x00,0x20,0x58,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  TTBR1_EL1, x0, x1
// CHECK: msrr  TTBR1_EL1, x0, x1           // encoding: [0x20,0x20,0x58,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  PAR_EL1, x0, x1
// CHECK: msrr  PAR_EL1, x0, x1             // encoding: [0x00,0x74,0x58,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  RCWSMASK_EL1, x0, x1
// CHECK: msrr  RCWSMASK_EL1, x0, x1        // encoding: [0x60,0xd0,0x58,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  RCWMASK_EL1, x0, x1
// CHECK: msrr  RCWMASK_EL1, x0, x1         // encoding: [0xc0,0xd0,0x58,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  TTBR0_EL2, x0, x1
// CHECK: msrr  TTBR0_EL2, x0, x1           // encoding: [0x00,0x20,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  TTBR1_EL2, x0, x1
// CHECK: msrr  TTBR1_EL2, x0, x1           // encoding: [0x20,0x20,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr  VTTBR_EL2, x0, x1
// CHECK: msrr  VTTBR_EL2, x0, x1           // encoding: [0x00,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          msrr   VTTBR_EL2, x0, x1
// CHECK: msrr   VTTBR_EL2, x0, x1           // encoding: [0x00,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x2, x3
// CHECK: msrr   VTTBR_EL2, x2, x3           // encoding: [0x02,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x4, x5
// CHECK: msrr   VTTBR_EL2, x4, x5           // encoding: [0x04,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x6, x7
// CHECK: msrr   VTTBR_EL2, x6, x7           // encoding: [0x06,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x8, x9
// CHECK: msrr   VTTBR_EL2, x8, x9           // encoding: [0x08,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x10, x11
// CHECK: msrr   VTTBR_EL2, x10, x11           // encoding: [0x0a,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x12, x13
// CHECK: msrr   VTTBR_EL2, x12, x13           // encoding: [0x0c,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x14, x15
// CHECK: msrr   VTTBR_EL2, x14, x15           // encoding: [0x0e,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x16, x17
// CHECK: msrr   VTTBR_EL2, x16, x17           // encoding: [0x10,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x18, x19
// CHECK: msrr   VTTBR_EL2, x18, x19           // encoding: [0x12,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x20, x21
// CHECK: msrr   VTTBR_EL2, x20, x21           // encoding: [0x14,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x22, x23
// CHECK: msrr   VTTBR_EL2, x22, x23           // encoding: [0x16,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x24, x25
// CHECK: msrr   VTTBR_EL2, x24, x25           // encoding: [0x18,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          msrr   VTTBR_EL2, x26, x27
// CHECK: msrr   VTTBR_EL2, x26, x27           // encoding: [0x1a,0x21,0x5c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          msrr TTBR0_EL1, x0, x2
// ERRORS: error: expected second odd register of a consecutive same-size even/odd register pair

          msrr TTBR0_EL1, x0
// ERRORS: error: expected comma

          msrr TTBR0_EL1, x1, x2
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          msrr TTBR0_EL1, x31, x0
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          msrr TTBR0_EL1, xzr, x30
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          msrr TTBR0_EL1, xzr
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          msrr S3_0_c2_c0_1
// ERRORS: error: too few operands for instruction

          msrr x0, x1, S3_0_c2_c0_1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair
