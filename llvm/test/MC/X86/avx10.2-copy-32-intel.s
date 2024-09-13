// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vmovd   xmm1, xmm2
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x7e,0xca]
          vmovd   xmm1, xmm2

// CHECK: vmovd   xmm1, xmm2
// CHECK: encoding: [0x62,0xf1,0x7d,0x08,0xd6,0xca]
          vmovd.s   xmm1, xmm2

// CHECK: vmovw   xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6e,0xca]
          vmovw   xmm1, xmm2

// CHECK: vmovw   xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x7e,0xca]
          vmovw.s   xmm1, xmm2
