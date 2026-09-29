// RUN: llvm-mc -triple i686 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      vsha512msg1 ymm2, xmm3
// CHECK: encoding: [0xc4,0xe2,0x7f,0xcc,0xd3]
               vsha512msg1 ymm2, xmm3

// CHECK:      vsha512msg2 ymm2, ymm3
// CHECK: encoding: [0xc4,0xe2,0x7f,0xcd,0xd3]
               vsha512msg2 ymm2, ymm3

// CHECK:      vsha512rnds2 ymm2, ymm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0x67,0xcb,0xd4]
               vsha512rnds2 ymm2, ymm3, xmm4
