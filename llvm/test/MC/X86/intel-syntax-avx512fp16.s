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
