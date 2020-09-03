// RUN:     llvm-mc -triple aarch64 -show-encoding  -mattr=+bf16 < %s       | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding  -mattr=+v8.6a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding  -mattr=-bf16  < %s 2>&1 | FileCheck %s --check-prefix=NOBF16
// RUN: not llvm-mc -triple aarch64 -show-encoding  < %s 2>&1 | FileCheck %s --check-prefix=NOBF16


bfdot v2.2s, v3.4h, v4.4h
bfdot v2.4s, v3.8h, v4.8h
// CHECK: bfdot v2.2s, v3.4h, v4.4h      // encoding: [0x62,0xfc,0x44,0x2e]
// CHECK: bfdot v2.4s, v3.8h, v4.8h      // encoding: [0x62,0xfc,0x44,0x6e]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.2s, v3.4h, v4.4h
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.4s, v3.8h, v4.8h

bfdot  v2.2s, v3.4h, v4.2h[0]
bfdot  v2.2s, v3.4h, v4.2h[1]
bfdot  v2.2s, v3.4h, v4.2h[2]
bfdot  v2.2s, v3.4h, v4.2h[3]
// CHECK: bfdot   v2.2s, v3.4h, v4.2h[0]  // encoding: [0x62,0xf0,0x44,0x0f]
// CHECK: bfdot   v2.2s, v3.4h, v4.2h[1]  // encoding: [0x62,0xf0,0x64,0x0f]
// CHECK: bfdot   v2.2s, v3.4h, v4.2h[2]  // encoding: [0x62,0xf8,0x44,0x0f]
// CHECK: bfdot   v2.2s, v3.4h, v4.2h[3]  // encoding: [0x62,0xf8,0x64,0x0f]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot   v2.2s, v3.4h, v4.2h[0]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot   v2.2s, v3.4h, v4.2h[1]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot   v2.2s, v3.4h, v4.2h[2]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot   v2.2s, v3.4h, v4.2h[3]


bfdot v2.4s, v3.8h, v4.2h[0]
bfdot v2.4s, v3.8h, v4.2h[1]
bfdot v2.4s, v3.8h, v4.2h[2]
bfdot v2.4s, v3.8h, v4.2h[3]
// CHECK: bfdot  v2.4s, v3.8h, v4.2h[0]  // encoding: [0x62,0xf0,0x44,0x4f]
// CHECK: bfdot  v2.4s, v3.8h, v4.2h[1]  // encoding: [0x62,0xf0,0x64,0x4f]
// CHECK: bfdot  v2.4s, v3.8h, v4.2h[2]  // encoding: [0x62,0xf8,0x44,0x4f]
// CHECK: bfdot  v2.4s, v3.8h, v4.2h[3]  // encoding: [0x62,0xf8,0x64,0x4f]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.4s, v3.8h, v4.2h[0]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.4s, v3.8h, v4.2h[1]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.4s, v3.8h, v4.2h[2]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfdot v2.4s, v3.8h, v4.2h[3]


bfmmla v2.4s, v3.8h, v4.8h
bfmmla v3.4s, v4.8h, v5.8h
// CHECK: bfmmla v2.4s, v3.8h, v4.8h   // encoding: [0x62,0xec,0x44,0x6e]
// CHECK: bfmmla v3.4s, v4.8h, v5.8h   // encoding: [0x83,0xec,0x45,0x6e]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfmmla v2.4s, v3.8h, v4.8h
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfmmla v3.4s, v4.8h, v5.8h

bfcvtn  v5.4h, v5.4s
bfcvtn2 v5.8h, v5.4s
// CHECK: bfcvtn  v5.4h, v5.4s           // encoding: [0xa5,0x68,0xa1,0x0e]
// CHECK: bfcvtn2 v5.8h, v5.4s           // encoding: [0xa5,0x68,0xa1,0x4e]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfcvtn  v5.4h, v5.4s
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfcvtn2 v5.8h, v5.4s

bfcvt  h5, s3
// CHECK: bfcvt   h5, s3               // encoding: [0x65,0x40,0x63,0x1e]
// NOBF16: instruction requires: bf16
// NOBF16-NEXT: bfcvt  h5, s3

bfmlalb V10.4S, V21.8h, V14.8H
bfmlalt V21.4S, V14.8h, V10.8H
// CHECK:       bfmlalb	v10.4s, v21.8h, v14.8h  // encoding: [0xaa,0xfe,0xce,0x2e]
// CHECK-NEXT:  bfmlalt	v21.4s, v14.8h, v10.8h  // encoding: [0xd5,0xfd,0xca,0x6e]
// NOBF16:      error: instruction requires: bf16
// NOBF16-NEXT: bfmlalb V10.4S, V21.8h, V14.8H
// NOBF16-NEXT: ^
// NOBF16:      instruction requires: bf16
// NOBF16-NEXT: bfmlalt V21.4S, V14.8h, V10.8H
// NOBF16-NEXT: ^

bfmlalb V14.4S, V21.8H, V10.H[1]
bfmlalb V14.4S, V21.8H, V10.H[2]
bfmlalb V14.4S, V21.8H, V10.H[7]
bfmlalt V21.4S, V10.8H, V14.H[1]
bfmlalt V21.4S, V10.8H, V14.H[2]
bfmlalt V21.4S, V10.8H, V14.H[7]
// CHECK:      bfmlalb v14.4s, v21.8h, v10.h[1] // encoding: [0xae,0xf2,0xda,0x0f]
// CHECK-NEXT: bfmlalb v14.4s, v21.8h, v10.h[2] // encoding: [0xae,0xf2,0xea,0x0f]
// CHECK-NEXT: bfmlalb v14.4s, v21.8h, v10.h[7] // encoding: [0xae,0xfa,0xfa,0x0f]
// CHECK-NEXT: bfmlalt v21.4s, v10.8h, v14.h[1] // encoding: [0x55,0xf1,0xde,0x4f]
// CHECK-NEXT: bfmlalt v21.4s, v10.8h, v14.h[2] // encoding: [0x55,0xf1,0xee,0x4f]
// CHECK-NEXT: bfmlalt v21.4s, v10.8h, v14.h[7] // encoding: [0x55,0xf9,0xfe,0x4f]
// NOBF16:      error: instruction requires: bf16
// NOBF16-NEXT: bfmlalb V14.4S, V21.8H, V10.H[1]
// NOBF16-NEXT: ^
// NOBF16:      error: instruction requires: bf16
// NOBF16-NEXT: bfmlalb V14.4S, V21.8H, V10.H[2]
// NOBF16-NEXT: ^
// NOBF16:      error: instruction requires: bf16
// NOBF16-NEXT: bfmlalb V14.4S, V21.8H, V10.H[7]
// NOBF16-NEXT: ^
// NOBF16:      instruction requires: bf16
// NOBF16-NEXT: bfmlalt V21.4S, V10.8H, V14.H[1]
// NOBF16-NEXT: ^
// NOBF16:      instruction requires: bf16
// NOBF16-NEXT: bfmlalt V21.4S, V10.8H, V14.H[2]
// NOBF16-NEXT: ^
// NOBF16:      instruction requires: bf16
// NOBF16-NEXT: bfmlalt V21.4S, V10.8H, V14.H[7]
// NOBF16-NEXT: ^
