// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: vmovd   %xmm22, %xmm21
// CHECK: encoding: [0x62,0xa1,0x7e,0x08,0x7e,0xee]
          vmovd   %xmm22, %xmm21

// CHECK: vmovd   %xmm22, %xmm21
// CHECK: encoding: [0x62,0xa1,0x7d,0x08,0xd6,0xee]
          vmovd.s   %xmm22, %xmm21

// CHECK: vmovw   %xmm22, %xmm21
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x6e,0xee]
          vmovw   %xmm22, %xmm21

// CHECK: vmovw   %xmm22, %xmm21
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x7e,0xee]
          vmovw.s   %xmm22, %xmm21
