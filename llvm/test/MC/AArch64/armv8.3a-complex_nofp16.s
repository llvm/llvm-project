// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+complxnum -o - %s 2>%t | FileCheck %s
.text
fcmla v0.2s, v1.2s, v2.2s, #0
fcmla v0.4s, v1.4s, v2.4s, #0
fcmla v0.2d, v1.2d, v2.2d, #0
fcmla v0.2s, v1.2s, v2.2s, #0
fcmla v0.2s, v1.2s, v2.2s, #90
fcmla v0.2s, v1.2s, v2.2s, #180
fcmla v0.2s, v1.2s, v2.2s, #270
fcadd v0.2s, v1.2s, v2.2s, #90
fcadd v0.4s, v1.4s, v2.4s, #90
fcadd v0.2d, v1.2d, v2.2d, #90
fcadd v0.2s, v1.2s, v2.2s, #90
fcadd v0.2s, v1.2s, v2.2s, #270
fcmla v0.4s, v1.4s, v2.s[0], #0
fcmla v0.4s, v1.4s, v2.s[0], #90
fcmla v0.4s, v1.4s, v2.s[0], #180
fcmla v0.4s, v1.4s, v2.s[0], #270
fcmla v0.4s, v1.4s, v2.s[1], #0
//CHECK: 	.text
//CHECK-NEXT: 	fcmla	v0.2s, v1.2s, v2.2s, #0 // encoding: [0x20,0xc4,0x82,0x2e]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.4s, #0 // encoding: [0x20,0xc4,0x82,0x6e]
//CHECK-NEXT: 	fcmla	v0.2d, v1.2d, v2.2d, #0 // encoding: [0x20,0xc4,0xc2,0x6e]
//CHECK-NEXT: 	fcmla	v0.2s, v1.2s, v2.2s, #0 // encoding: [0x20,0xc4,0x82,0x2e]
//CHECK-NEXT: 	fcmla	v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xcc,0x82,0x2e]
//CHECK-NEXT: 	fcmla	v0.2s, v1.2s, v2.2s, #180 // encoding: [0x20,0xd4,0x82,0x2e]
//CHECK-NEXT: 	fcmla	v0.2s, v1.2s, v2.2s, #270 // encoding: [0x20,0xdc,0x82,0x2e]
//CHECK-NEXT: 	fcadd	v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xe4,0x82,0x2e]
//CHECK-NEXT: 	fcadd	v0.4s, v1.4s, v2.4s, #90 // encoding: [0x20,0xe4,0x82,0x6e]
//CHECK-NEXT: 	fcadd	v0.2d, v1.2d, v2.2d, #90 // encoding: [0x20,0xe4,0xc2,0x6e]
//CHECK-NEXT: 	fcadd	v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xe4,0x82,0x2e]
//CHECK-NEXT: 	fcadd	v0.2s, v1.2s, v2.2s, #270 // encoding: [0x20,0xf4,0x82,0x2e]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.s[0], #0 // encoding: [0x20,0x10,0x82,0x6f]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.s[0], #90 // encoding: [0x20,0x30,0x82,0x6f]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.s[0], #180 // encoding: [0x20,0x50,0x82,0x6f]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.s[0], #270 // encoding: [0x20,0x70,0x82,0x6f]
//CHECK-NEXT: 	fcmla	v0.4s, v1.4s, v2.s[1], #0 // encoding: [0x20,0x18,0x82,0x6f]

