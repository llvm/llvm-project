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

// CHECK: vcmpneq_usph k5, zmm5, zmm4
// CHECK: encoding: [0x62,0xf3,0x54,0x48,0xc2,0xec,0x14]
          vcmpneq_usph k5, zmm5, zmm4

// CHECK: vcmpnlt_uqph k5, zmm5, zmm4, {sae}
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0xc2,0xec,0x15]
          vcmpnlt_uqph k5, zmm5, zmm4, {sae}

// CHECK: vcmpnle_uqph k5 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf3,0x54,0x4f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x16]
          vcmpnle_uqph k5 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcmpord_sph k5, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf3,0x54,0x58,0xc2,0x29,0x17]
          vcmpord_sph k5, zmm5, word ptr [ecx]{1to32}

// CHECK: vcmpeq_usph k5, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf3,0x54,0x48,0xc2,0x69,0x7f,0x18]
          vcmpeq_usph k5, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vcmpnge_uqph k5 {k7}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf3,0x54,0x5f,0xc2,0x6a,0x80,0x19]
          vcmpnge_uqph k5 {k7}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vcmpngt_uqsh k5, xmm5, xmm4
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0xec,0x1a]
          vcmpngt_uqsh k5, xmm5, xmm4

// CHECK: vcmpfalse_ossh k5, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf3,0x56,0x18,0xc2,0xec,0x1b]
          vcmpfalse_ossh k5, xmm5, xmm4, {sae}

// CHECK: vcmpneq_ossh k5 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf3,0x56,0x0f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x1c]
          vcmpneq_ossh k5 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vcmpge_oqsh k5, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0x29,0x1d]
          vcmpge_oqsh k5, xmm5, word ptr [ecx]

// CHECK: vcmpgt_oqsh k5, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf3,0x56,0x08,0xc2,0x69,0x7f,0x1e]
          vcmpgt_oqsh k5, xmm5, word ptr [ecx + 254]

// CHECK: vcmptrue_ussh k5 {k7}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf3,0x56,0x0f,0xc2,0x6a,0x80,0x1f]
          vcmptrue_ussh k5 {k7}, xmm5, word ptr [edx - 256]

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

// CHECK: vcvtdq2ph ymm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x5b,0xf5]
          vcvtdq2ph ymm6, zmm5

// CHECK: vcvtdq2ph ymm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x5b,0xf5]
          vcvtdq2ph ymm6, zmm5, {rn-sae}

// CHECK: vcvtdq2ph ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtdq2ph ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtdq2ph ymm6, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x5b,0x31]
          vcvtdq2ph ymm6, dword ptr [ecx]{1to16}

// CHECK: vcvtdq2ph ymm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x5b,0x71,0x7f]
          vcvtdq2ph ymm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtdq2ph ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x5b,0x72,0x80]
          vcvtdq2ph ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvtpd2ph xmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x5a,0xf5]
          vcvtpd2ph xmm6, zmm5

// CHECK: vcvtpd2ph xmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x5a,0xf5]
          vcvtpd2ph xmm6, zmm5, {rn-sae}

// CHECK: vcvtpd2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtpd2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtpd2ph xmm6, qword ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0x58,0x5a,0x31]
          vcvtpd2ph xmm6, qword ptr [ecx]{1to8}

// CHECK: vcvtpd2ph xmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x5a,0x71,0x7f]
          vcvtpd2ph xmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtpd2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0xdf,0x5a,0x72,0x80]
          vcvtpd2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvtph2dq zmm6, ymm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x5b,0xf5]
          vcvtph2dq zmm6, ymm5

// CHECK: vcvtph2dq zmm6, ymm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x5b,0xf5]
          vcvtph2dq zmm6, ymm5, {rn-sae}

// CHECK: vcvtph2dq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2dq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2dq zmm6, word ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x5b,0x31]
          vcvtph2dq zmm6, word ptr [ecx]{1to16}

// CHECK: vcvtph2dq zmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x5b,0x71,0x7f]
          vcvtph2dq zmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2dq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x5b,0x72,0x80]
          vcvtph2dq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2pd zmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x5a,0xf5]
          vcvtph2pd zmm6, xmm5

// CHECK: vcvtph2pd zmm6, xmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x5a,0xf5]
          vcvtph2pd zmm6, xmm5, {sae}

// CHECK: vcvtph2pd zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2pd zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2pd zmm6, word ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x5a,0x31]
          vcvtph2pd zmm6, word ptr [ecx]{1to8}

// CHECK: vcvtph2pd zmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x5a,0x71,0x7f]
          vcvtph2pd zmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2pd zmm6 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x5a,0x72,0x80]
          vcvtph2pd zmm6 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2psx zmm6, ymm5
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x13,0xf5]
          vcvtph2psx zmm6, ymm5

// CHECK: vcvtph2psx zmm6, ymm5, {sae}
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x13,0xf5]
          vcvtph2psx zmm6, ymm5, {sae}

// CHECK: vcvtph2psx zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7d,0x4f,0x13,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2psx zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2psx zmm6, word ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7d,0x58,0x13,0x31]
          vcvtph2psx zmm6, word ptr [ecx]{1to16}

// CHECK: vcvtph2psx zmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x13,0x71,0x7f]
          vcvtph2psx zmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2psx zmm6 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf6,0x7d,0xdf,0x13,0x72,0x80]
          vcvtph2psx zmm6 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2qq zmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7b,0xf5]
          vcvtph2qq zmm6, xmm5

// CHECK: vcvtph2qq zmm6, xmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7b,0xf5]
          vcvtph2qq zmm6, xmm5, {rn-sae}

// CHECK: vcvtph2qq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x7b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2qq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2qq zmm6, word ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x7b,0x31]
          vcvtph2qq zmm6, word ptr [ecx]{1to8}

// CHECK: vcvtph2qq zmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7b,0x71,0x7f]
          vcvtph2qq zmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2qq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x7b,0x72,0x80]
          vcvtph2qq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2udq zmm6, ymm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x79,0xf5]
          vcvtph2udq zmm6, ymm5

// CHECK: vcvtph2udq zmm6, ymm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x79,0xf5]
          vcvtph2udq zmm6, ymm5, {rn-sae}

// CHECK: vcvtph2udq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2udq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2udq zmm6, word ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x79,0x31]
          vcvtph2udq zmm6, word ptr [ecx]{1to16}

// CHECK: vcvtph2udq zmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x79,0x71,0x7f]
          vcvtph2udq zmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvtph2udq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x79,0x72,0x80]
          vcvtph2udq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvtph2uqq zmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x79,0xf5]
          vcvtph2uqq zmm6, xmm5

// CHECK: vcvtph2uqq zmm6, xmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x79,0xf5]
          vcvtph2uqq zmm6, xmm5, {rn-sae}

// CHECK: vcvtph2uqq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uqq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2uqq zmm6, word ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x79,0x31]
          vcvtph2uqq zmm6, word ptr [ecx]{1to8}

// CHECK: vcvtph2uqq zmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x79,0x71,0x7f]
          vcvtph2uqq zmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvtph2uqq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x79,0x72,0x80]
          vcvtph2uqq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvtph2uw zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x7d,0xf5]
          vcvtph2uw zmm6, zmm5

// CHECK: vcvtph2uw zmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x7d,0xf5]
          vcvtph2uw zmm6, zmm5, {rn-sae}

// CHECK: vcvtph2uw zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uw zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2uw zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x7d,0x31]
          vcvtph2uw zmm6, word ptr [ecx]{1to32}

// CHECK: vcvtph2uw zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x7d,0x71,0x7f]
          vcvtph2uw zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2uw zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x7d,0x72,0x80]
          vcvtph2uw zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtph2w zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7d,0xf5]
          vcvtph2w zmm6, zmm5

// CHECK: vcvtph2w zmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7d,0xf5]
          vcvtph2w zmm6, zmm5, {rn-sae}

// CHECK: vcvtph2w zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2w zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtph2w zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x7d,0x31]
          vcvtph2w zmm6, word ptr [ecx]{1to32}

// CHECK: vcvtph2w zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7d,0x71,0x7f]
          vcvtph2w zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtph2w zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x7d,0x72,0x80]
          vcvtph2w zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtps2phx ymm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x1d,0xf5]
          vcvtps2phx ymm6, zmm5

// CHECK: vcvtps2phx ymm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x1d,0xf5]
          vcvtps2phx ymm6, zmm5, {rn-sae}

// CHECK: vcvtps2phx ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x1d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2phx ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtps2phx ymm6, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x1d,0x31]
          vcvtps2phx ymm6, dword ptr [ecx]{1to16}

// CHECK: vcvtps2phx ymm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x1d,0x71,0x7f]
          vcvtps2phx ymm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtps2phx ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x1d,0x72,0x80]
          vcvtps2phx ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvtqq2ph xmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x5b,0xf5]
          vcvtqq2ph xmm6, zmm5

// CHECK: vcvtqq2ph xmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x5b,0xf5]
          vcvtqq2ph xmm6, zmm5, {rn-sae}

// CHECK: vcvtqq2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtqq2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtqq2ph xmm6, qword ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0x58,0x5b,0x31]
          vcvtqq2ph xmm6, qword ptr [ecx]{1to8}

// CHECK: vcvtqq2ph xmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x5b,0x71,0x7f]
          vcvtqq2ph xmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtqq2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0xdf,0x5b,0x72,0x80]
          vcvtqq2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvtsd2sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0xd7,0x08,0x5a,0xf4]
          vcvtsd2sh xmm6, xmm5, xmm4

// CHECK: vcvtsd2sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xd7,0x18,0x5a,0xf4]
          vcvtsd2sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vcvtsd2sh xmm6 {k7}, xmm5, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xd7,0x0f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtsd2sh xmm6 {k7}, xmm5, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsd2sh xmm6, xmm5, qword ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0xd7,0x08,0x5a,0x31]
          vcvtsd2sh xmm6, xmm5, qword ptr [ecx]

// CHECK: vcvtsd2sh xmm6, xmm5, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0xd7,0x08,0x5a,0x71,0x7f]
          vcvtsd2sh xmm6, xmm5, qword ptr [ecx + 1016]

// CHECK: vcvtsd2sh xmm6 {k7} {z}, xmm5, qword ptr [edx - 1024]
// CHECK: encoding: [0x62,0xf5,0xd7,0x8f,0x5a,0x72,0x80]
          vcvtsd2sh xmm6 {k7} {z}, xmm5, qword ptr [edx - 1024]

// CHECK: vcvtsh2sd xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5a,0xf4]
          vcvtsh2sd xmm6, xmm5, xmm4

// CHECK: vcvtsh2sd xmm6, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x5a,0xf4]
          vcvtsh2sd xmm6, xmm5, xmm4, {sae}

// CHECK: vcvtsh2sd xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtsh2sd xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsh2sd xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5a,0x31]
          vcvtsh2sd xmm6, xmm5, word ptr [ecx]

// CHECK: vcvtsh2sd xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x5a,0x71,0x7f]
          vcvtsh2sd xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vcvtsh2sd xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x5a,0x72,0x80]
          vcvtsh2sd xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vcvtsh2si edx, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0xd6]
          vcvtsh2si edx, xmm6

// CHECK: vcvtsh2si edx, xmm6, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x2d,0xd6]
          vcvtsh2si edx, xmm6, {rn-sae}

// CHECK: vcvtsh2si edx, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtsh2si edx, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsh2si edx, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x11]
          vcvtsh2si edx, word ptr [ecx]

// CHECK: vcvtsh2si edx, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x51,0x7f]
          vcvtsh2si edx, word ptr [ecx + 254]

// CHECK: vcvtsh2si edx, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x52,0x80]
          vcvtsh2si edx, word ptr [edx - 256]

// CHECK: vcvtsh2ss xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x54,0x08,0x13,0xf4]
          vcvtsh2ss xmm6, xmm5, xmm4

// CHECK: vcvtsh2ss xmm6, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf6,0x54,0x18,0x13,0xf4]
          vcvtsh2ss xmm6, xmm5, xmm4, {sae}

// CHECK: vcvtsh2ss xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x54,0x0f,0x13,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtsh2ss xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsh2ss xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x54,0x08,0x13,0x31]
          vcvtsh2ss xmm6, xmm5, word ptr [ecx]

// CHECK: vcvtsh2ss xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x54,0x08,0x13,0x71,0x7f]
          vcvtsh2ss xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vcvtsh2ss xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x54,0x8f,0x13,0x72,0x80]
          vcvtsh2ss xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vcvtsh2usi edx, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0xd6]
          vcvtsh2usi edx, xmm6

// CHECK: vcvtsh2usi edx, xmm6, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x79,0xd6]
          vcvtsh2usi edx, xmm6, {rn-sae}

// CHECK: vcvtsh2usi edx, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtsh2usi edx, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsh2usi edx, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x11]
          vcvtsh2usi edx, word ptr [ecx]

// CHECK: vcvtsh2usi edx, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x51,0x7f]
          vcvtsh2usi edx, word ptr [ecx + 254]

// CHECK: vcvtsh2usi edx, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x52,0x80]
          vcvtsh2usi edx, word ptr [edx - 256]

// CHECK: vcvtsi2sh xmm6, xmm5, edx
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x2a,0xf2]
          vcvtsi2sh xmm6, xmm5, edx

// CHECK: vcvtsi2sh xmm6, xmm5, {rn-sae}, edx
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x2a,0xf2]
          vcvtsi2sh xmm6, xmm5, {rn-sae}, edx

// CHECK: vcvtsi2sh xmm6, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x2a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtsi2sh xmm6, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtsi2sh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x2a,0x31]
          vcvtsi2sh xmm6, xmm5, dword ptr [ecx]

// CHECK: vcvtsi2sh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x2a,0x71,0x7f]
          vcvtsi2sh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vcvtsi2sh xmm6, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x2a,0x72,0x80]
          vcvtsi2sh xmm6, xmm5, dword ptr [edx - 512]

// CHECK: vcvtss2sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x1d,0xf4]
          vcvtss2sh xmm6, xmm5, xmm4

// CHECK: vcvtss2sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x1d,0xf4]
          vcvtss2sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vcvtss2sh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x1d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtss2sh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtss2sh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x1d,0x31]
          vcvtss2sh xmm6, xmm5, dword ptr [ecx]

// CHECK: vcvtss2sh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x1d,0x71,0x7f]
          vcvtss2sh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vcvtss2sh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf5,0x54,0x8f,0x1d,0x72,0x80]
          vcvtss2sh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]

// CHECK: vcvttph2dq zmm6, ymm5
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x5b,0xf5]
          vcvttph2dq zmm6, ymm5

// CHECK: vcvttph2dq zmm6, ymm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x5b,0xf5]
          vcvttph2dq zmm6, ymm5, {sae}

// CHECK: vcvttph2dq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2dq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2dq zmm6, word ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x5b,0x31]
          vcvttph2dq zmm6, word ptr [ecx]{1to16}

// CHECK: vcvttph2dq zmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x5b,0x71,0x7f]
          vcvttph2dq zmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvttph2dq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x5b,0x72,0x80]
          vcvttph2dq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttph2qq zmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7a,0xf5]
          vcvttph2qq zmm6, xmm5

// CHECK: vcvttph2qq zmm6, xmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7a,0xf5]
          vcvttph2qq zmm6, xmm5, {sae}

// CHECK: vcvttph2qq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2qq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2qq zmm6, word ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x7a,0x31]
          vcvttph2qq zmm6, word ptr [ecx]{1to8}

// CHECK: vcvttph2qq zmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7a,0x71,0x7f]
          vcvttph2qq zmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvttph2qq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x7a,0x72,0x80]
          vcvttph2qq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttph2udq zmm6, ymm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x78,0xf5]
          vcvttph2udq zmm6, ymm5

// CHECK: vcvttph2udq zmm6, ymm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x78,0xf5]
          vcvttph2udq zmm6, ymm5, {sae}

// CHECK: vcvttph2udq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2udq zmm6 {k7}, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2udq zmm6, word ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x78,0x31]
          vcvttph2udq zmm6, word ptr [ecx]{1to16}

// CHECK: vcvttph2udq zmm6, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x78,0x71,0x7f]
          vcvttph2udq zmm6, ymmword ptr [ecx + 4064]

// CHECK: vcvttph2udq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x78,0x72,0x80]
          vcvttph2udq zmm6 {k7} {z}, word ptr [edx - 256]{1to16}

// CHECK: vcvttph2uqq zmm6, xmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x78,0xf5]
          vcvttph2uqq zmm6, xmm5

// CHECK: vcvttph2uqq zmm6, xmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x78,0xf5]
          vcvttph2uqq zmm6, xmm5, {sae}

// CHECK: vcvttph2uqq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uqq zmm6 {k7}, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2uqq zmm6, word ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x78,0x31]
          vcvttph2uqq zmm6, word ptr [ecx]{1to8}

// CHECK: vcvttph2uqq zmm6, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x78,0x71,0x7f]
          vcvttph2uqq zmm6, xmmword ptr [ecx + 2032]

// CHECK: vcvttph2uqq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x78,0x72,0x80]
          vcvttph2uqq zmm6 {k7} {z}, word ptr [edx - 256]{1to8}

// CHECK: vcvttph2uw zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x7c,0xf5]
          vcvttph2uw zmm6, zmm5

// CHECK: vcvttph2uw zmm6, zmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x7c,0xf5]
          vcvttph2uw zmm6, zmm5, {sae}

// CHECK: vcvttph2uw zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uw zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2uw zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x7c,0x31]
          vcvttph2uw zmm6, word ptr [ecx]{1to32}

// CHECK: vcvttph2uw zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x7c,0x71,0x7f]
          vcvttph2uw zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvttph2uw zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x7c,0x72,0x80]
          vcvttph2uw zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttph2w zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7c,0xf5]
          vcvttph2w zmm6, zmm5

// CHECK: vcvttph2w zmm6, zmm5, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7c,0xf5]
          vcvttph2w zmm6, zmm5, {sae}

// CHECK: vcvttph2w zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2w zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttph2w zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x7c,0x31]
          vcvttph2w zmm6, word ptr [ecx]{1to32}

// CHECK: vcvttph2w zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x7c,0x71,0x7f]
          vcvttph2w zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvttph2w zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x7c,0x72,0x80]
          vcvttph2w zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvttsh2si edx, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0xd6]
          vcvttsh2si edx, xmm6

// CHECK: vcvttsh2si edx, xmm6, {sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x2c,0xd6]
          vcvttsh2si edx, xmm6, {sae}

// CHECK: vcvttsh2si edx, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttsh2si edx, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvttsh2si edx, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x11]
          vcvttsh2si edx, word ptr [ecx]

// CHECK: vcvttsh2si edx, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x51,0x7f]
          vcvttsh2si edx, word ptr [ecx + 254]

// CHECK: vcvttsh2si edx, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x52,0x80]
          vcvttsh2si edx, word ptr [edx - 256]

// CHECK: vcvttsh2usi edx, xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0xd6]
          vcvttsh2usi edx, xmm6

// CHECK: vcvttsh2usi edx, xmm6, {sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x78,0xd6]
          vcvttsh2usi edx, xmm6, {sae}

// CHECK: vcvttsh2usi edx, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttsh2usi edx, word ptr [esp + 8*esi + 268435456]

// CHECK: vcvttsh2usi edx, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x11]
          vcvttsh2usi edx, word ptr [ecx]

// CHECK: vcvttsh2usi edx, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x51,0x7f]
          vcvttsh2usi edx, word ptr [ecx + 254]

// CHECK: vcvttsh2usi edx, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x52,0x80]
          vcvttsh2usi edx, word ptr [edx - 256]

// CHECK: vcvtudq2ph ymm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x7a,0xf5]
          vcvtudq2ph ymm6, zmm5

// CHECK: vcvtudq2ph ymm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x7a,0xf5]
          vcvtudq2ph ymm6, zmm5, {rn-sae}

// CHECK: vcvtudq2ph ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtudq2ph ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtudq2ph ymm6, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x7a,0x31]
          vcvtudq2ph ymm6, dword ptr [ecx]{1to16}

// CHECK: vcvtudq2ph ymm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x7a,0x71,0x7f]
          vcvtudq2ph ymm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtudq2ph ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x7a,0x72,0x80]
          vcvtudq2ph ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvtuqq2ph xmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0xff,0x48,0x7a,0xf5]
          vcvtuqq2ph xmm6, zmm5

// CHECK: vcvtuqq2ph xmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xff,0x18,0x7a,0xf5]
          vcvtuqq2ph xmm6, zmm5, {rn-sae}

// CHECK: vcvtuqq2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xff,0x4f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtuqq2ph xmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtuqq2ph xmm6, qword ptr [ecx]{1to8}
// CHECK: encoding: [0x62,0xf5,0xff,0x58,0x7a,0x31]
          vcvtuqq2ph xmm6, qword ptr [ecx]{1to8}

// CHECK: vcvtuqq2ph xmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xff,0x48,0x7a,0x71,0x7f]
          vcvtuqq2ph xmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtuqq2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xff,0xdf,0x7a,0x72,0x80]
          vcvtuqq2ph xmm6 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvtusi2sh xmm6, xmm5, edx
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x7b,0xf2]
          vcvtusi2sh xmm6, xmm5, edx

// CHECK: vcvtusi2sh xmm6, xmm5, {rn-sae}, edx
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x7b,0xf2]
          vcvtusi2sh xmm6, xmm5, {rn-sae}, edx

// CHECK: vcvtusi2sh xmm6, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x7b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtusi2sh xmm6, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtusi2sh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x7b,0x31]
          vcvtusi2sh xmm6, xmm5, dword ptr [ecx]

// CHECK: vcvtusi2sh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x7b,0x71,0x7f]
          vcvtusi2sh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vcvtusi2sh xmm6, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x7b,0x72,0x80]
          vcvtusi2sh xmm6, xmm5, dword ptr [edx - 512]

// CHECK: vcvtuw2ph zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x7d,0xf5]
          vcvtuw2ph zmm6, zmm5

// CHECK: vcvtuw2ph zmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x7d,0xf5]
          vcvtuw2ph zmm6, zmm5, {rn-sae}

// CHECK: vcvtuw2ph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtuw2ph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtuw2ph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x7d,0x31]
          vcvtuw2ph zmm6, word ptr [ecx]{1to32}

// CHECK: vcvtuw2ph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x7d,0x71,0x7f]
          vcvtuw2ph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtuw2ph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x7d,0x72,0x80]
          vcvtuw2ph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vcvtw2ph zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x7d,0xf5]
          vcvtw2ph zmm6, zmm5

// CHECK: vcvtw2ph zmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x7d,0xf5]
          vcvtw2ph zmm6, zmm5, {rn-sae}

// CHECK: vcvtw2ph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtw2ph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtw2ph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x7d,0x31]
          vcvtw2ph zmm6, word ptr [ecx]{1to32}

// CHECK: vcvtw2ph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x7d,0x71,0x7f]
          vcvtw2ph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtw2ph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x7d,0x72,0x80]
          vcvtw2ph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vfpclassph k5, zmm6, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x66,0xee,0x7b]
          vfpclassph k5, zmm6, 123

// CHECK: vfpclassph k5 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x4f,0x66,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclassph k5 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vfpclassph k5, word ptr [ecx]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x58,0x66,0x29,0x7b]
          vfpclassph k5, word ptr [ecx]{1to32}, 123

// CHECK: vfpclassph k5, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x66,0x69,0x7f,0x7b]
          vfpclassph k5, zmmword ptr [ecx + 8128], 123

// CHECK: vfpclassph k5 {k7}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassph k5 {k7}, word ptr [edx - 256]{1to32}, 123

// CHECK: vfpclasssh k5, xmm6, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x67,0xee,0x7b]
          vfpclasssh k5, xmm6, 123

// CHECK: vfpclasssh k5 {k7}, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x67,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclasssh k5 {k7}, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vfpclasssh k5, word ptr [ecx], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x67,0x29,0x7b]
          vfpclasssh k5, word ptr [ecx], 123

// CHECK: vfpclasssh k5, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x67,0x69,0x7f,0x7b]
          vfpclasssh k5, word ptr [ecx + 254], 123

// CHECK: vfpclasssh k5 {k7}, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x67,0x6a,0x80,0x7b]
          vfpclasssh k5 {k7}, word ptr [edx - 256], 123

// CHECK: vgetexpph zmm6, zmm5
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x42,0xf5]
          vgetexpph zmm6, zmm5

// CHECK: vgetexpph zmm6, zmm5, {sae}
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x42,0xf5]
          vgetexpph zmm6, zmm5, {sae}

// CHECK: vgetexpph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7d,0x4f,0x42,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vgetexpph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vgetexpph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0x58,0x42,0x31]
          vgetexpph zmm6, word ptr [ecx]{1to32}

// CHECK: vgetexpph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x42,0x71,0x7f]
          vgetexpph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vgetexpph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0xdf,0x42,0x72,0x80]
          vgetexpph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vgetexpsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x43,0xf4]
          vgetexpsh xmm6, xmm5, xmm4

// CHECK: vgetexpsh xmm6, xmm5, xmm4, {sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x43,0xf4]
          vgetexpsh xmm6, xmm5, xmm4, {sae}

// CHECK: vgetexpsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x43,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vgetexpsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vgetexpsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x43,0x31]
          vgetexpsh xmm6, xmm5, word ptr [ecx]

// CHECK: vgetexpsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x43,0x71,0x7f]
          vgetexpsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vgetexpsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x43,0x72,0x80]
          vgetexpsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vgetmantph zmm6, zmm5, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x26,0xf5,0x7b]
          vgetmantph zmm6, zmm5, 123

// CHECK: vgetmantph zmm6, zmm5, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x26,0xf5,0x7b]
          vgetmantph zmm6, zmm5, {sae}, 123

// CHECK: vgetmantph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x4f,0x26,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantph zmm6, word ptr [ecx]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x58,0x26,0x31,0x7b]
          vgetmantph zmm6, word ptr [ecx]{1to32}, 123

// CHECK: vgetmantph zmm6, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x26,0x71,0x7f,0x7b]
          vgetmantph zmm6, zmmword ptr [ecx + 8128], 123

// CHECK: vgetmantph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vgetmantsh xmm6, xmm5, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x27,0xf4,0x7b]
          vgetmantsh xmm6, xmm5, xmm4, 123

// CHECK: vgetmantsh xmm6, xmm5, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0x27,0xf4,0x7b]
          vgetmantsh xmm6, xmm5, xmm4, {sae}, 123

// CHECK: vgetmantsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x0f,0x27,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vgetmantsh xmm6, xmm5, word ptr [ecx], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x27,0x31,0x7b]
          vgetmantsh xmm6, xmm5, word ptr [ecx], 123

// CHECK: vgetmantsh xmm6, xmm5, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x27,0x71,0x7f,0x7b]
          vgetmantsh xmm6, xmm5, word ptr [ecx + 254], 123

// CHECK: vgetmantsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x8f,0x27,0x72,0x80,0x7b]
          vgetmantsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123

// CHECK: vrcpph zmm6, zmm5
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x4c,0xf5]
          vrcpph zmm6, zmm5

// CHECK: vrcpph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7d,0x4f,0x4c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrcpph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrcpph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0x58,0x4c,0x31]
          vrcpph zmm6, word ptr [ecx]{1to32}

// CHECK: vrcpph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x4c,0x71,0x7f]
          vrcpph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vrcpph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0xdf,0x4c,0x72,0x80]
          vrcpph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vrcpsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4d,0xf4]
          vrcpsh xmm6, xmm5, xmm4

// CHECK: vrcpsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x4d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrcpsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vrcpsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4d,0x31]
          vrcpsh xmm6, xmm5, word ptr [ecx]

// CHECK: vrcpsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4d,0x71,0x7f]
          vrcpsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vrcpsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x4d,0x72,0x80]
          vrcpsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vreduceph zmm6, zmm5, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x56,0xf5,0x7b]
          vreduceph zmm6, zmm5, 123

// CHECK: vreduceph zmm6, zmm5, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x56,0xf5,0x7b]
          vreduceph zmm6, zmm5, {sae}, 123

// CHECK: vreduceph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x4f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreduceph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vreduceph zmm6, word ptr [ecx]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x58,0x56,0x31,0x7b]
          vreduceph zmm6, word ptr [ecx]{1to32}, 123

// CHECK: vreduceph zmm6, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x56,0x71,0x7f,0x7b]
          vreduceph zmm6, zmmword ptr [ecx + 8128], 123

// CHECK: vreduceph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0xdf,0x56,0x72,0x80,0x7b]
          vreduceph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vreducesh xmm6, xmm5, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x57,0xf4,0x7b]
          vreducesh xmm6, xmm5, xmm4, 123

// CHECK: vreducesh xmm6, xmm5, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0x57,0xf4,0x7b]
          vreducesh xmm6, xmm5, xmm4, {sae}, 123

// CHECK: vreducesh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x0f,0x57,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducesh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vreducesh xmm6, xmm5, word ptr [ecx], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x57,0x31,0x7b]
          vreducesh xmm6, xmm5, word ptr [ecx], 123

// CHECK: vreducesh xmm6, xmm5, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x57,0x71,0x7f,0x7b]
          vreducesh xmm6, xmm5, word ptr [ecx + 254], 123

// CHECK: vreducesh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x8f,0x57,0x72,0x80,0x7b]
          vreducesh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123

// CHECK: vrndscaleph zmm6, zmm5, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x08,0xf5,0x7b]
          vrndscaleph zmm6, zmm5, 123

// CHECK: vrndscaleph zmm6, zmm5, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x08,0xf5,0x7b]
          vrndscaleph zmm6, zmm5, {sae}, 123

// CHECK: vrndscaleph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x4f,0x08,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscaleph zmm6, word ptr [ecx]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x58,0x08,0x31,0x7b]
          vrndscaleph zmm6, word ptr [ecx]{1to32}, 123

// CHECK: vrndscaleph zmm6, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x08,0x71,0x7f,0x7b]
          vrndscaleph zmm6, zmmword ptr [ecx + 8128], 123

// CHECK: vrndscaleph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0xdf,0x08,0x72,0x80,0x7b]
          vrndscaleph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}, 123

// CHECK: vrndscalesh xmm6, xmm5, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x0a,0xf4,0x7b]
          vrndscalesh xmm6, xmm5, xmm4, 123

// CHECK: vrndscalesh xmm6, xmm5, xmm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0x0a,0xf4,0x7b]
          vrndscalesh xmm6, xmm5, xmm4, {sae}, 123

// CHECK: vrndscalesh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x0f,0x0a,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalesh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456], 123

// CHECK: vrndscalesh xmm6, xmm5, word ptr [ecx], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x0a,0x31,0x7b]
          vrndscalesh xmm6, xmm5, word ptr [ecx], 123

// CHECK: vrndscalesh xmm6, xmm5, word ptr [ecx + 254], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0x0a,0x71,0x7f,0x7b]
          vrndscalesh xmm6, xmm5, word ptr [ecx + 254], 123

// CHECK: vrndscalesh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123
// CHECK: encoding: [0x62,0xf3,0x54,0x8f,0x0a,0x72,0x80,0x7b]
          vrndscalesh xmm6 {k7} {z}, xmm5, word ptr [edx - 256], 123

// CHECK: vrsqrtph zmm6, zmm5
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x4e,0xf5]
          vrsqrtph zmm6, zmm5

// CHECK: vrsqrtph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x7d,0x4f,0x4e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0x58,0x4e,0x31]
          vrsqrtph zmm6, word ptr [ecx]{1to32}

// CHECK: vrsqrtph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x7d,0x48,0x4e,0x71,0x7f]
          vrsqrtph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vrsqrtph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x7d,0xdf,0x4e,0x72,0x80]
          vrsqrtph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vrsqrtsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4f,0xf4]
          vrsqrtsh xmm6, xmm5, xmm4

// CHECK: vrsqrtsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x4f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vrsqrtsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4f,0x31]
          vrsqrtsh xmm6, xmm5, word ptr [ecx]

// CHECK: vrsqrtsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x4f,0x71,0x7f]
          vrsqrtsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vrsqrtsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x4f,0x72,0x80]
          vrsqrtsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vscalefph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x2c,0xf4]
          vscalefph zmm6, zmm5, zmm4

// CHECK: vscalefph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x2c,0xf4]
          vscalefph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vscalefph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x2c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vscalefph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vscalefph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x2c,0x31]
          vscalefph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vscalefph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x2c,0x71,0x7f]
          vscalefph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vscalefph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x2c,0x72,0x80]
          vscalefph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vscalefsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x2d,0xf4]
          vscalefsh xmm6, xmm5, xmm4

// CHECK: vscalefsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x2d,0xf4]
          vscalefsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vscalefsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x2d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vscalefsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vscalefsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x2d,0x31]
          vscalefsh xmm6, xmm5, word ptr [ecx]

// CHECK: vscalefsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x2d,0x71,0x7f]
          vscalefsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vscalefsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x2d,0x72,0x80]
          vscalefsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vsqrtph zmm6, zmm5
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x51,0xf5]
          vsqrtph zmm6, zmm5

// CHECK: vsqrtph zmm6, zmm5, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x51,0xf5]
          vsqrtph zmm6, zmm5, {rn-sae}

// CHECK: vsqrtph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsqrtph zmm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtph zmm6, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x51,0x31]
          vsqrtph zmm6, word ptr [ecx]{1to32}

// CHECK: vsqrtph zmm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x51,0x71,0x7f]
          vsqrtph zmm6, zmmword ptr [ecx + 8128]

// CHECK: vsqrtph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x51,0x72,0x80]
          vsqrtph zmm6 {k7} {z}, word ptr [edx - 256]{1to32}

// CHECK: vsqrtsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x51,0xf4]
          vsqrtsh xmm6, xmm5, xmm4

// CHECK: vsqrtsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x56,0x18,0x51,0xf4]
          vsqrtsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vsqrtsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x56,0x0f,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsqrtsh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vsqrtsh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x51,0x31]
          vsqrtsh xmm6, xmm5, word ptr [ecx]

// CHECK: vsqrtsh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf5,0x56,0x08,0x51,0x71,0x7f]
          vsqrtsh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vsqrtsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf5,0x56,0x8f,0x51,0x72,0x80]
          vsqrtsh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmadd132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x98,0xf4]
          vfmadd132ph zmm6, zmm5, zmm4

// CHECK: vfmadd132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x98,0xf4]
          vfmadd132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x98,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x98,0x31]
          vfmadd132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x98,0x71,0x7f]
          vfmadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x98,0x72,0x80]
          vfmadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmadd132sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x99,0xf4]
          vfmadd132sh xmm6, xmm5, xmm4

// CHECK: vfmadd132sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x99,0xf4]
          vfmadd132sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmadd132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x99,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd132sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x99,0x31]
          vfmadd132sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmadd132sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x99,0x71,0x7f]
          vfmadd132sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmadd132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x99,0x72,0x80]
          vfmadd132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmadd213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa8,0xf4]
          vfmadd213ph zmm6, zmm5, zmm4

// CHECK: vfmadd213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa8,0xf4]
          vfmadd213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xa8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xa8,0x31]
          vfmadd213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa8,0x71,0x7f]
          vfmadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xa8,0x72,0x80]
          vfmadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmadd213sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa9,0xf4]
          vfmadd213sh xmm6, xmm5, xmm4

// CHECK: vfmadd213sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa9,0xf4]
          vfmadd213sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmadd213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xa9,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd213sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa9,0x31]
          vfmadd213sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmadd213sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa9,0x71,0x7f]
          vfmadd213sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmadd213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xa9,0x72,0x80]
          vfmadd213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmadd231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb8,0xf4]
          vfmadd231ph zmm6, zmm5, zmm4

// CHECK: vfmadd231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb8,0xf4]
          vfmadd231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xb8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xb8,0x31]
          vfmadd231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb8,0x71,0x7f]
          vfmadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xb8,0x72,0x80]
          vfmadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmadd231sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb9,0xf4]
          vfmadd231sh xmm6, xmm5, xmm4

// CHECK: vfmadd231sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb9,0xf4]
          vfmadd231sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmadd231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xb9,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmadd231sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb9,0x31]
          vfmadd231sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmadd231sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb9,0x71,0x7f]
          vfmadd231sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmadd231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xb9,0x72,0x80]
          vfmadd231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmaddsub132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x96,0xf4]
          vfmaddsub132ph zmm6, zmm5, zmm4

// CHECK: vfmaddsub132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x96,0xf4]
          vfmaddsub132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmaddsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x96,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmaddsub132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x96,0x31]
          vfmaddsub132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmaddsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x96,0x71,0x7f]
          vfmaddsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmaddsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x96,0x72,0x80]
          vfmaddsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmaddsub213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa6,0xf4]
          vfmaddsub213ph zmm6, zmm5, zmm4

// CHECK: vfmaddsub213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa6,0xf4]
          vfmaddsub213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmaddsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xa6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmaddsub213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xa6,0x31]
          vfmaddsub213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmaddsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa6,0x71,0x7f]
          vfmaddsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmaddsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xa6,0x72,0x80]
          vfmaddsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmaddsub231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb6,0xf4]
          vfmaddsub231ph zmm6, zmm5, zmm4

// CHECK: vfmaddsub231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb6,0xf4]
          vfmaddsub231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmaddsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xb6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmaddsub231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xb6,0x31]
          vfmaddsub231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmaddsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb6,0x71,0x7f]
          vfmaddsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmaddsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xb6,0x72,0x80]
          vfmaddsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsub132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9a,0xf4]
          vfmsub132ph zmm6, zmm5, zmm4

// CHECK: vfmsub132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9a,0xf4]
          vfmsub132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x9a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x9a,0x31]
          vfmsub132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9a,0x71,0x7f]
          vfmsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x9a,0x72,0x80]
          vfmsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsub132sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9b,0xf4]
          vfmsub132sh xmm6, xmm5, xmm4

// CHECK: vfmsub132sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9b,0xf4]
          vfmsub132sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmsub132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub132sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9b,0x31]
          vfmsub132sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmsub132sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9b,0x71,0x7f]
          vfmsub132sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmsub132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x9b,0x72,0x80]
          vfmsub132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmsub213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xaa,0xf4]
          vfmsub213ph zmm6, zmm5, zmm4

// CHECK: vfmsub213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xaa,0xf4]
          vfmsub213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xaa,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xaa,0x31]
          vfmsub213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xaa,0x71,0x7f]
          vfmsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xaa,0x72,0x80]
          vfmsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsub213sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xab,0xf4]
          vfmsub213sh xmm6, xmm5, xmm4

// CHECK: vfmsub213sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xab,0xf4]
          vfmsub213sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmsub213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xab,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub213sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xab,0x31]
          vfmsub213sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmsub213sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xab,0x71,0x7f]
          vfmsub213sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmsub213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xab,0x72,0x80]
          vfmsub213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmsub231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xba,0xf4]
          vfmsub231ph zmm6, zmm5, zmm4

// CHECK: vfmsub231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xba,0xf4]
          vfmsub231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xba,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xba,0x31]
          vfmsub231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xba,0x71,0x7f]
          vfmsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xba,0x72,0x80]
          vfmsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsub231sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbb,0xf4]
          vfmsub231sh xmm6, xmm5, xmm4

// CHECK: vfmsub231sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbb,0xf4]
          vfmsub231sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmsub231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xbb,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfmsub231sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbb,0x31]
          vfmsub231sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfmsub231sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbb,0x71,0x7f]
          vfmsub231sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfmsub231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xbb,0x72,0x80]
          vfmsub231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfmsubadd132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x97,0xf4]
          vfmsubadd132ph zmm6, zmm5, zmm4

// CHECK: vfmsubadd132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x97,0xf4]
          vfmsubadd132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsubadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x97,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsubadd132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x97,0x31]
          vfmsubadd132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsubadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x97,0x71,0x7f]
          vfmsubadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsubadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x97,0x72,0x80]
          vfmsubadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsubadd213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa7,0xf4]
          vfmsubadd213ph zmm6, zmm5, zmm4

// CHECK: vfmsubadd213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa7,0xf4]
          vfmsubadd213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsubadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xa7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsubadd213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xa7,0x31]
          vfmsubadd213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsubadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xa7,0x71,0x7f]
          vfmsubadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsubadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xa7,0x72,0x80]
          vfmsubadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfmsubadd231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb7,0xf4]
          vfmsubadd231ph zmm6, zmm5, zmm4

// CHECK: vfmsubadd231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb7,0xf4]
          vfmsubadd231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmsubadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xb7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmsubadd231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xb7,0x31]
          vfmsubadd231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfmsubadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xb7,0x71,0x7f]
          vfmsubadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmsubadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xb7,0x72,0x80]
          vfmsubadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9c,0xf4]
          vfnmadd132ph zmm6, zmm5, zmm4

// CHECK: vfnmadd132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9c,0xf4]
          vfnmadd132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x9c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x9c,0x31]
          vfnmadd132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9c,0x71,0x7f]
          vfnmadd132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x9c,0x72,0x80]
          vfnmadd132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd132sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9d,0xf4]
          vfnmadd132sh xmm6, xmm5, xmm4

// CHECK: vfnmadd132sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9d,0xf4]
          vfnmadd132sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmadd132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd132sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9d,0x31]
          vfnmadd132sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmadd132sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9d,0x71,0x7f]
          vfnmadd132sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmadd132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x9d,0x72,0x80]
          vfnmadd132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfnmadd213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xac,0xf4]
          vfnmadd213ph zmm6, zmm5, zmm4

// CHECK: vfnmadd213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xac,0xf4]
          vfnmadd213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xac,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xac,0x31]
          vfnmadd213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xac,0x71,0x7f]
          vfnmadd213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xac,0x72,0x80]
          vfnmadd213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd213sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xad,0xf4]
          vfnmadd213sh xmm6, xmm5, xmm4

// CHECK: vfnmadd213sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xad,0xf4]
          vfnmadd213sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmadd213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xad,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd213sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xad,0x31]
          vfnmadd213sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmadd213sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xad,0x71,0x7f]
          vfnmadd213sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmadd213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xad,0x72,0x80]
          vfnmadd213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfnmadd231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xbc,0xf4]
          vfnmadd231ph zmm6, zmm5, zmm4

// CHECK: vfnmadd231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbc,0xf4]
          vfnmadd231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xbc,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xbc,0x31]
          vfnmadd231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xbc,0x71,0x7f]
          vfnmadd231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xbc,0x72,0x80]
          vfnmadd231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmadd231sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbd,0xf4]
          vfnmadd231sh xmm6, xmm5, xmm4

// CHECK: vfnmadd231sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbd,0xf4]
          vfnmadd231sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmadd231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xbd,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmadd231sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbd,0x31]
          vfnmadd231sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmadd231sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbd,0x71,0x7f]
          vfnmadd231sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmadd231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xbd,0x72,0x80]
          vfnmadd231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfnmsub132ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9e,0xf4]
          vfnmsub132ph zmm6, zmm5, zmm4

// CHECK: vfnmsub132ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9e,0xf4]
          vfnmsub132ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0x9e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0x9e,0x31]
          vfnmsub132ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0x9e,0x71,0x7f]
          vfnmsub132ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0x9e,0x72,0x80]
          vfnmsub132ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub132sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9f,0xf4]
          vfnmsub132sh xmm6, xmm5, xmm4

// CHECK: vfnmsub132sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9f,0xf4]
          vfnmsub132sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmsub132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub132sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9f,0x31]
          vfnmsub132sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmsub132sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9f,0x71,0x7f]
          vfnmsub132sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmsub132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0x9f,0x72,0x80]
          vfnmsub132sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfnmsub213ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xae,0xf4]
          vfnmsub213ph zmm6, zmm5, zmm4

// CHECK: vfnmsub213ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xae,0xf4]
          vfnmsub213ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xae,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xae,0x31]
          vfnmsub213ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xae,0x71,0x7f]
          vfnmsub213ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xae,0x72,0x80]
          vfnmsub213ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub213sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xaf,0xf4]
          vfnmsub213sh xmm6, xmm5, xmm4

// CHECK: vfnmsub213sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xaf,0xf4]
          vfnmsub213sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmsub213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xaf,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub213sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xaf,0x31]
          vfnmsub213sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmsub213sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xaf,0x71,0x7f]
          vfnmsub213sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmsub213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xaf,0x72,0x80]
          vfnmsub213sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfnmsub231ph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xbe,0xf4]
          vfnmsub231ph zmm6, zmm5, zmm4

// CHECK: vfnmsub231ph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbe,0xf4]
          vfnmsub231ph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfnmsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x4f,0xbe,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231ph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231ph zmm6, zmm5, word ptr [ecx]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0x58,0xbe,0x31]
          vfnmsub231ph zmm6, zmm5, word ptr [ecx]{1to32}

// CHECK: vfnmsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x55,0x48,0xbe,0x71,0x7f]
          vfnmsub231ph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfnmsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}
// CHECK: encoding: [0x62,0xf6,0x55,0xdf,0xbe,0x72,0x80]
          vfnmsub231ph zmm6 {k7} {z}, zmm5, word ptr [edx - 256]{1to32}

// CHECK: vfnmsub231sh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbf,0xf4]
          vfnmsub231sh xmm6, xmm5, xmm4

// CHECK: vfnmsub231sh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbf,0xf4]
          vfnmsub231sh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfnmsub231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xbf,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231sh xmm6 {k7}, xmm5, word ptr [esp + 8*esi + 268435456]

// CHECK: vfnmsub231sh xmm6, xmm5, word ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbf,0x31]
          vfnmsub231sh xmm6, xmm5, word ptr [ecx]

// CHECK: vfnmsub231sh xmm6, xmm5, word ptr [ecx + 254]
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbf,0x71,0x7f]
          vfnmsub231sh xmm6, xmm5, word ptr [ecx + 254]

// CHECK: vfnmsub231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]
// CHECK: encoding: [0x62,0xf6,0x55,0x8f,0xbf,0x72,0x80]
          vfnmsub231sh xmm6 {k7} {z}, xmm5, word ptr [edx - 256]

// CHECK: vfcmaddcph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x57,0x48,0x56,0xf4]
          vfcmaddcph zmm6, zmm5, zmm4

// CHECK: vfcmaddcph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0x56,0xf4]
          vfcmaddcph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfcmaddcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x57,0x4f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmaddcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfcmaddcph zmm6, zmm5, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf6,0x57,0x58,0x56,0x31]
          vfcmaddcph zmm6, zmm5, dword ptr [ecx]{1to16}

// CHECK: vfcmaddcph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x57,0x48,0x56,0x71,0x7f]
          vfcmaddcph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfcmaddcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf6,0x57,0xdf,0x56,0x72,0x80]
          vfcmaddcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}

// CHECK: vfcmaddcsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0x57,0xf4]
          vfcmaddcsh xmm6, xmm5, xmm4

// CHECK: vfcmaddcsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0x57,0xf4]
          vfcmaddcsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfcmaddcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x57,0x0f,0x57,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmaddcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vfcmaddcsh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0x57,0x31]
          vfcmaddcsh xmm6, xmm5, dword ptr [ecx]

// CHECK: vfcmaddcsh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0x57,0x71,0x7f]
          vfcmaddcsh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vfcmaddcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf6,0x57,0x8f,0x57,0x72,0x80]
          vfcmaddcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]

// CHECK: vfcmulcph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x57,0x48,0xd6,0xf4]
          vfcmulcph zmm6, zmm5, zmm4

// CHECK: vfcmulcph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0xd6,0xf4]
          vfcmulcph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfcmulcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x57,0x4f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmulcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfcmulcph zmm6, zmm5, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf6,0x57,0x58,0xd6,0x31]
          vfcmulcph zmm6, zmm5, dword ptr [ecx]{1to16}

// CHECK: vfcmulcph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x57,0x48,0xd6,0x71,0x7f]
          vfcmulcph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfcmulcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf6,0x57,0xdf,0xd6,0x72,0x80]
          vfcmulcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}

// CHECK: vfcmulcsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0xd7,0xf4]
          vfcmulcsh xmm6, xmm5, xmm4

// CHECK: vfcmulcsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0xd7,0xf4]
          vfcmulcsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfcmulcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x57,0x0f,0xd7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmulcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vfcmulcsh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0xd7,0x31]
          vfcmulcsh xmm6, xmm5, dword ptr [ecx]

// CHECK: vfcmulcsh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0xd7,0x71,0x7f]
          vfcmulcsh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vfcmulcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf6,0x57,0x8f,0xd7,0x72,0x80]
          vfcmulcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]

// CHECK: vfmaddcph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x56,0x48,0x56,0xf4]
          vfmaddcph zmm6, zmm5, zmm4

// CHECK: vfmaddcph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0x56,0xf4]
          vfmaddcph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmaddcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x56,0x4f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmaddcph zmm6, zmm5, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf6,0x56,0x58,0x56,0x31]
          vfmaddcph zmm6, zmm5, dword ptr [ecx]{1to16}

// CHECK: vfmaddcph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x56,0x48,0x56,0x71,0x7f]
          vfmaddcph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmaddcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf6,0x56,0xdf,0x56,0x72,0x80]
          vfmaddcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}

// CHECK: vfmaddcsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0x57,0xf4]
          vfmaddcsh xmm6, xmm5, xmm4

// CHECK: vfmaddcsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0x57,0xf4]
          vfmaddcsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmaddcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x56,0x0f,0x57,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vfmaddcsh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0x57,0x31]
          vfmaddcsh xmm6, xmm5, dword ptr [ecx]

// CHECK: vfmaddcsh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0x57,0x71,0x7f]
          vfmaddcsh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vfmaddcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf6,0x56,0x8f,0x57,0x72,0x80]
          vfmaddcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]

// CHECK: vfmulcph zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf6,0x56,0x48,0xd6,0xf4]
          vfmulcph zmm6, zmm5, zmm4

// CHECK: vfmulcph zmm6, zmm5, zmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0xd6,0xf4]
          vfmulcph zmm6, zmm5, zmm4, {rn-sae}

// CHECK: vfmulcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x56,0x4f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmulcph zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vfmulcph zmm6, zmm5, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf6,0x56,0x58,0xd6,0x31]
          vfmulcph zmm6, zmm5, dword ptr [ecx]{1to16}

// CHECK: vfmulcph zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf6,0x56,0x48,0xd6,0x71,0x7f]
          vfmulcph zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vfmulcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf6,0x56,0xdf,0xd6,0x72,0x80]
          vfmulcph zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}

// CHECK: vfmulcsh xmm6, xmm5, xmm4
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0xd7,0xf4]
          vfmulcsh xmm6, xmm5, xmm4

// CHECK: vfmulcsh xmm6, xmm5, xmm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0xd7,0xf4]
          vfmulcsh xmm6, xmm5, xmm4, {rn-sae}

// CHECK: vfmulcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf6,0x56,0x0f,0xd7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmulcsh xmm6 {k7}, xmm5, dword ptr [esp + 8*esi + 268435456]

// CHECK: vfmulcsh xmm6, xmm5, dword ptr [ecx]
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0xd7,0x31]
          vfmulcsh xmm6, xmm5, dword ptr [ecx]

// CHECK: vfmulcsh xmm6, xmm5, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0xd7,0x71,0x7f]
          vfmulcsh xmm6, xmm5, dword ptr [ecx + 508]

// CHECK: vfmulcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf6,0x56,0x8f,0xd7,0x72,0x80]
          vfmulcsh xmm6 {k7} {z}, xmm5, dword ptr [edx - 512]
