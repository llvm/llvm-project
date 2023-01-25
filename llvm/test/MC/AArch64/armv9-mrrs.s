// +the required for RCWSMASK_EL1, RCWMASK_EL1
// +el2vmsa required for TTBR0_EL2 (VSCTLR_EL2), VTTBR_EL2
// +vh required for TTBR1_EL2

// RUN: not llvm-mc -triple aarch64 -mattr=+d128,+the,+el2vmsa,+vh -show-encoding %s -o - 2> %t | FileCheck %s
// RUN: FileCheck %s --input-file=%t --check-prefix=ERRORS

// RUN: not llvm-mc -triple aarch64 -mattr=+the,+el2vmsa,+vh -show-encoding %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR-NO-D128

          mrrs  x0, x1, TTBR0_EL1
// CHECK: mrrs  x0, x1, TTBR0_EL1           // encoding: [0x00,0x20,0x78,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, TTBR1_EL1
// CHECK: mrrs  x0, x1, TTBR1_EL1           // encoding: [0x20,0x20,0x78,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, PAR_EL1
// CHECK: mrrs  x0, x1, PAR_EL1             // encoding: [0x00,0x74,0x78,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, RCWSMASK_EL1
// CHECK: mrrs  x0, x1, RCWSMASK_EL1        // encoding: [0x60,0xd0,0x78,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, RCWMASK_EL1
// CHECK: mrrs  x0, x1, RCWMASK_EL1         // encoding: [0xc0,0xd0,0x78,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, TTBR0_EL2
// CHECK: mrrs  x0, x1, TTBR0_EL2           // encoding: [0x00,0x20,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, TTBR1_EL2
// CHECK: mrrs  x0, x1, TTBR1_EL2           // encoding: [0x20,0x20,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x0, x1, VTTBR_EL2
// CHECK: mrrs  x0, x1, VTTBR_EL2           // encoding: [0x00,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          mrrs   x0,  x1, VTTBR_EL2
// CHECK: mrrs   x0,  x1, VTTBR_EL2           // encoding: [0x00,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs   x2,  x3, VTTBR_EL2
// CHECK: mrrs   x2,  x3, VTTBR_EL2           // encoding: [0x02,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs   x4,  x5, VTTBR_EL2
// CHECK: mrrs   x4,  x5, VTTBR_EL2           // encoding: [0x04,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs   x6,  x7, VTTBR_EL2
// CHECK: mrrs   x6,  x7, VTTBR_EL2           // encoding: [0x06,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs   x8,  x9, VTTBR_EL2
// CHECK: mrrs   x8,  x9, VTTBR_EL2           // encoding: [0x08,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x10, x11, VTTBR_EL2
// CHECK: mrrs  x10, x11, VTTBR_EL2           // encoding: [0x0a,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x12, x13, VTTBR_EL2
// CHECK: mrrs  x12, x13, VTTBR_EL2           // encoding: [0x0c,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x14, x15, VTTBR_EL2
// CHECK: mrrs  x14, x15, VTTBR_EL2           // encoding: [0x0e,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x16, x17, VTTBR_EL2
// CHECK: mrrs  x16, x17, VTTBR_EL2           // encoding: [0x10,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x18, x19, VTTBR_EL2
// CHECK: mrrs  x18, x19, VTTBR_EL2           // encoding: [0x12,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x20, x21, VTTBR_EL2
// CHECK: mrrs  x20, x21, VTTBR_EL2           // encoding: [0x14,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x22, x23, VTTBR_EL2
// CHECK: mrrs  x22, x23, VTTBR_EL2           // encoding: [0x16,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x24, x25, VTTBR_EL2
// CHECK: mrrs  x24, x25, VTTBR_EL2           // encoding: [0x18,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          mrrs  x26, x27, VTTBR_EL2
// CHECK: mrrs  x26, x27, VTTBR_EL2           // encoding: [0x1a,0x21,0x7c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          mrrs x0, x2, TTBR0_EL1
// ERRORS: error: expected second odd register of a consecutive same-size even/odd register pair

          mrrs x0, TTBR0_EL1
// ERRORS: error: expected second odd register of a consecutive same-size even/odd register pair

          mrrs x1, x2, TTBR0_EL1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          mrrs x31, x0, TTBR0_EL1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          mrrs xzr, x30, TTBR0_EL1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          mrrs xzr, TTBR0_EL1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          mrrs S3_0_c2_c0_1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair

          mrrs S3_0_c2_c0_1, x0, x1
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair
