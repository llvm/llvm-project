// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vmovd   %xmm2, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x7e,0xca]
          vmovd   %xmm2, %xmm1

// CHECK: vmovd   %xmm2, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7d,0x08,0xd6,0xca]
          vmovd.s   %xmm2, %xmm1

// CHECK: vmovw   %xmm2, %xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6e,0xca]
          vmovw   %xmm2, %xmm1

// CHECK: vmovw   %xmm2, %xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x7e,0xca]
          vmovw.s   %xmm2, %xmm1
