// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vsha512msg1 %xmm3, %ymm12
// CHECK: encoding: [0xc4,0x62,0x7f,0xcc,0xe3]
          vsha512msg1 %xmm3, %ymm12

// CHECK: vsha512msg2 %ymm3, %ymm12
// CHECK: encoding: [0xc4,0x62,0x7f,0xcd,0xe3]
          vsha512msg2 %ymm3, %ymm12

// CHECK: vsha512rnds2 %xmm4, %ymm3, %ymm12
// CHECK: encoding: [0xc4,0x62,0x67,0xcb,0xe4]
          vsha512rnds2 %xmm4, %ymm3, %ymm12

