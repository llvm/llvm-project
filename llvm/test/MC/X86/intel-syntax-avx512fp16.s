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

// CHECK: vaddph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x58,0xf4]
          vaddph zmm6, zmm5, zmm4

// CHECK: vaddph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x58,0xf4]
          vaddph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vaddph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x58,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vaddph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vaddph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x58,0x31]
          vaddph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vaddph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x58,0x71,0x7f]
          vaddph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vaddph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x58,0x72,0x80]
          vaddph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vaddsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x58,0xf4]
          vaddsh xmm6, xmm5, xmm4

// CHECK: vaddsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x58,0xf4]
          vaddsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vaddsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x58,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vaddsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vaddsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x58,0x31]
          vaddsh xmm6, xmm5, word ptr [ecx]

// CHECK: vaddsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x58,0x71,0x7f]
          vaddsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vaddsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x58,0x72,0x80]
          vaddsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vcmpph k5, zmm5, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x48,0xc2,0xec,0x7b]
          vcmpph k5, zmm5, zmm4, 123

// CHECK: vcmpph k5, zmm5, zmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0xc2,0xec,0x7b]
          vcmpph k5, zmm5, zmm4, {sae}, 123

// CHECK: vcmpph k5 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x4f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpph k5 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmpph k5, zmm5, word ptr [ecx]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x58,0xc2,0x29,0x7b]
          vcmpph k5, zmm5, word ptr [ecx]{1to32}, 123

// CHECK: vcmpph k5, zmm5, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x48,0xc2,0x69,0x7f,0x7b]
          vcmpph k5, zmm5, zmmword ptr [ecx + 8128], 123

// CHECK: vcmpph k5 {k7}, zmm5, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x5f,0xc2,0x6a,0x80,0x7b]
          vcmpph k5 {k7}, zmm5, word ptr [edx - 256]{1to32}, 123

// CHECK: vcmpsh k5, xmm5, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0xec,0x7b]
          vcmpsh k5, xmm5, xmm4, 123

// CHECK: vcmpsh k5, xmm5, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x56,0x18,0xc2,0xec,0x7b]
          vcmpsh k5, xmm5, xmm4, {sae}, 123

// CHECK: vcmpsh k5 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x56,0x0f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpsh k5 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vcmpsh k5, xmm5, word ptr [ecx], 123
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0x29,0x7b]
          vcmpsh k5, xmm5, word ptr [ecx], 123

// CHECK: vcmpsh k5, xmm5, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0x69,0x7f,0x7b]
          vcmpsh k5, xmm5, word ptr [ecx + 254], 123

// CHECK: vcmpsh k5 {k7}, xmm5, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x56,0x0f,0xc2,0x6a,0x80,0x7b]
          vcmpsh k5 {k7}, xmm5, word ptr [edx - 256], 123

// CHECK: vcomish xmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2f,0xf5]
          vcomish xmm6, xmm5

// CHECK: vcomish xmm6, xmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x2f,0xf5]
          vcomish xmm6, xmm5, {sae}

// CHECK: vcomish xmm6, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcomish xmm6, word ptr [esp + 8*esi + 268435456]

// CHECK: vcomish xmm6, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2f,0x31]
          vcomish xmm6, word ptr [ecx]

// CHECK: vcomish xmm6, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2f,0x71,0x7f]
          vcomish xmm6, word ptr [ecx + 254]

// CHECK: vcomish xmm6, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2f,0x72,0x80]
          vcomish xmm6, word ptr [edx - 256]

// CHECK: vdivph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5e,0xf4]
          vdivph zmm6, zmm5, zmm4

// CHECK: vdivph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5e,0xf4]
          vdivph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vdivph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x5e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdivph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdivph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x5e,0x31]
          vdivph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vdivph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5e,0x71,0x7f]
          vdivph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vdivph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x5e,0x72,0x80]
          vdivph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vdivsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5e,0xf4]
          vdivsh xmm6, xmm5, xmm4

// CHECK: vdivsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x5e,0xf4]
          vdivsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vdivsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x5e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdivsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vdivsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5e,0x31]
          vdivsh xmm6, xmm5, word ptr [ecx]

// CHECK: vdivsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5e,0x71,0x7f]
          vdivsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vdivsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x5e,0x72,0x80]
          vdivsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vmaxph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5f,0xf4]
          vmaxph zmm6, zmm5, zmm4

// CHECK: vmaxph zmm6, zmm5, zmm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5f,0xf4]
          vmaxph zmm6, zmm5, zmm4, {sae}

// CHECK: vmaxph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x5f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmaxph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmaxph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x5f,0x31]
          vmaxph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vmaxph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5f,0x71,0x7f]
          vmaxph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vmaxph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x5f,0x72,0x80]
          vmaxph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vmaxsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5f,0xf4]
          vmaxsh xmm6, xmm5, xmm4

// CHECK: vmaxsh xmm6, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x5f,0xf4]
          vmaxsh xmm6, xmm5, xmm4, {sae}

// CHECK: vmaxsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x5f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmaxsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vmaxsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5f,0x31]
          vmaxsh xmm6, xmm5, word ptr [ecx]

// CHECK: vmaxsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5f,0x71,0x7f]
          vmaxsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vmaxsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x5f,0x72,0x80]
          vmaxsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vminph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5d,0xf4]
          vminph zmm6, zmm5, zmm4

// CHECK: vminph zmm6, zmm5, zmm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5d,0xf4]
          vminph zmm6, zmm5, zmm4, {sae}

// CHECK: vminph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x5d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vminph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vminph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x5d,0x31]
          vminph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vminph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5d,0x71,0x7f]
          vminph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vminph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x5d,0x72,0x80]
          vminph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vminsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5d,0xf4]
          vminsh xmm6, xmm5, xmm4

// CHECK: vminsh xmm6, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x5d,0xf4]
          vminsh xmm6, xmm5, xmm4, {sae}

// CHECK: vminsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x5d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vminsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vminsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5d,0x31]
          vminsh xmm6, xmm5, word ptr [ecx]

// CHECK: vminsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5d,0x71,0x7f]
          vminsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vminsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x5d,0x72,0x80]
          vminsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vmulph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x59,0xf4]
          vmulph zmm6, zmm5, zmm4

// CHECK: vmulph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x59,0xf4]
          vmulph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vmulph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x59,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmulph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vmulph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x59,0x31]
          vmulph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vmulph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x59,0x71,0x7f]
          vmulph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vmulph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x59,0x72,0x80]
          vmulph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vmulsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x59,0xf4]
          vmulsh xmm6, xmm5, xmm4

// CHECK: vmulsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x59,0xf4]
          vmulsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vmulsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x59,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmulsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vmulsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x59,0x31]
          vmulsh xmm6, xmm5, word ptr [ecx]

// CHECK: vmulsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x59,0x71,0x7f]
          vmulsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vmulsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x59,0x72,0x80]
          vmulsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vsubph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5c,0xf4]
          vsubph zmm6, zmm5, zmm4

// CHECK: vsubph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5c,0xf4]
          vsubph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vsubph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x4f,0x5c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsubph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsubph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0x58,0x5c,0x31]
          vsubph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vsubph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x54,0x48,0x5c,0x71,0x7f]
          vsubph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vsubph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x54,0xdf,0x5c,0x72,0x80]
          vsubph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vsubsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5c,0xf4]
          vsubsh xmm6, xmm5, xmm4

// CHECK: vsubsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x5c,0xf4]
          vsubsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vsubsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x5c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsubsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vsubsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5c,0x31]
          vsubsh xmm6, xmm5, word ptr [ecx]

// CHECK: vsubsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5c,0x71,0x7f]
          vsubsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vsubsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x5c,0x72,0x80]
          vsubsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vucomish xmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2e,0xf5]
          vucomish xmm6, xmm5

// CHECK: vucomish xmm6, xmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x2e,0xf5]
          vucomish xmm6, xmm5, {sae}

// CHECK: vucomish xmm6, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vucomish xmm6, word ptr [esp + 8*esi + 268435456]

// CHECK: vucomish xmm6, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2e,0x31]
          vucomish xmm6, word ptr [ecx]

// CHECK: vucomish xmm6, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2e,0x71,0x7f]
          vucomish xmm6, word ptr [ecx + 254]

// CHECK: vucomish xmm6, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x2e,0x72,0x80]
          vucomish xmm6, word ptr [edx - 256]
