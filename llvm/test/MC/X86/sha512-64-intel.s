// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vsha512msg1 ymm12, xmm3
// CHECK: encoding: [0xc4,0x62,0x7f,0xcc,0xe3]
          vsha512msg1 ymm12, xmm3

// CHECK: vsha512msg2 ymm12, ymm3
// CHECK: encoding: [0xc4,0x62,0x7f,0xcd,0xe3]
          vsha512msg2 ymm12, ymm3

// CHECK: vsha512rnds2 ymm12, ymm3, xmm4
// CHECK: encoding: [0xc4,0x62,0x67,0xcb,0xe4]
          vsha512rnds2 ymm12, ymm3, xmm4

