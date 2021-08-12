// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vmovsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x10,0xf4]
          vmovsh xmm6, xmm5, xmm4

// CHECK: vmovsh xmm6 {k7}, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x10,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmovsh xmm6 {k7}, word ptr [esp + 8*esi + 268435456]

// CHECK: vmovsh xmm6, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x10,0x31]
          vmovsh xmm6, word ptr [ecx]

// CHECK: vmovsh xmm6, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x10,0x71,0x7f]
          vmovsh xmm6, word ptr [ecx + 254]

// CHECK: vmovsh xmm6 {k7} {z}, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x10,0x72,0x80]
          vmovsh xmm6 {k7} {z}, word ptr [edx - 256]

// CHECK: vmovsh word ptr [esp + 8*esi + 268435456] {k7}, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x11,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmovsh word ptr [esp + 8*esi + 268435456] {k7}, xmm6

// CHECK: vmovsh word ptr [ecx], xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x11,0x31]
          vmovsh word ptr [ecx], xmm6

// CHECK: vmovsh word ptr [ecx + 254], xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x11,0x71,0x7f]
          vmovsh word ptr [ecx + 254], xmm6

// CHECK: vmovsh word ptr [edx - 256] {k7}, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x11,0x72,0x80]
          vmovsh word ptr [edx - 256] {k7}, xmm6

// CHECK: vmovw xmm6, edx
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6e,0xf2]
          vmovw xmm6, edx

// CHECK: vmovw edx, xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7e,0xf2]
          vmovw edx, xmm6

// CHECK: vmovw xmm6, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmovw xmm6, word ptr [esp + 8*esi + 268435456]

// CHECK: vmovw xmm6, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6e,0x31]
          vmovw xmm6, word ptr [ecx]

// CHECK: vmovw xmm6, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6e,0x71,0x7f]
          vmovw xmm6, word ptr [ecx + 254]

// CHECK: vmovw xmm6, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6e,0x72,0x80]
          vmovw xmm6, word ptr [edx - 256]

// CHECK: vmovw word ptr [esp + 8*esi + 268435456], xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmovw word ptr [esp + 8*esi + 268435456], xmm6

// CHECK: vmovw word ptr [ecx], xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7e,0x31]
          vmovw word ptr [ecx], xmm6

// CHECK: vmovw word ptr [ecx + 254], xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7e,0x71,0x7f]
          vmovw word ptr [ecx + 254], xmm6

// CHECK: vmovw word ptr [edx - 256], xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7e,0x72,0x80]
          vmovw word ptr [edx - 256], xmm6
