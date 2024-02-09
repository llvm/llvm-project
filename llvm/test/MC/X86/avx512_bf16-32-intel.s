// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xf4]
          vcvtne2ps2bf16 zmm6, zmm5, zmm4

// CHECK: vcvtne2ps2bf16 zmm6 {k7}, zmm5, zmm4
// CHECK: encoding: [0x62,0xf2,0x57,0x4f,0x72,0xf4]
          vcvtne2ps2bf16 zmm6 {k7}, zmm5, zmm4

// CHECK: vcvtne2ps2bf16 zmm6 {k7} {z}, zmm5, zmm4
// CHECK: encoding: [0x62,0xf2,0x57,0xcf,0x72,0xf4]
          vcvtne2ps2bf16 zmm6 {k7} {z}, zmm5, zmm4

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [ecx]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x31]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [ecx]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp + 8*esi + 291]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp + 8*esi + 291]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp - 4]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xb4,0x24,0xfc,0xff,0xff,0xff]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [esp - 4]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x30]
          vcvtne2ps2bf16 zmm6, zmm5, dword ptr [eax]{1to16}

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [edx + 8128]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x72,0x7f]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [edx + 8128]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [edx - 8192]
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x72,0x80]
          vcvtne2ps2bf16 zmm6, zmm5, zmmword ptr [edx - 8192]

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, dword ptr [edx + 508]{1to16}
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x72,0x7f]
          vcvtne2ps2bf16 zmm6, zmm5, dword ptr [edx + 508]{1to16}

// CHECK: vcvtne2ps2bf16 zmm6, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x72,0x80]
          vcvtne2ps2bf16 zmm6, zmm5, dword ptr [edx - 512]{1to16}

// CHECK: vcvtneps2bf16 ymm6, zmm5
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x72,0xf5]
          vcvtneps2bf16 ymm6, zmm5

// CHECK: vcvtneps2bf16 ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16 ymm6 {k7}, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvtneps2bf16 ymm6, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf2,0x7e,0x58,0x72,0x31]
          vcvtneps2bf16 ymm6, dword ptr [ecx]{1to16}

// CHECK: vcvtneps2bf16 ymm6, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x72,0x71,0x7f]
          vcvtneps2bf16 ymm6, zmmword ptr [ecx + 8128]

// CHECK: vcvtneps2bf16 ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x7e,0xdf,0x72,0x72,0x80]
          vcvtneps2bf16 ymm6 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vdpbf16ps zmm6, zmm5, zmm4
// CHECK: encoding: [0x62,0xf2,0x56,0x48,0x52,0xf4]
          vdpbf16ps zmm6, zmm5, zmm4

// CHECK: vdpbf16ps zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x56,0x4f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps zmm6 {k7}, zmm5, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vdpbf16ps zmm6, zmm5, dword ptr [ecx]{1to16}
// CHECK: encoding: [0x62,0xf2,0x56,0x58,0x52,0x31]
          vdpbf16ps zmm6, zmm5, dword ptr [ecx]{1to16}

// CHECK: vdpbf16ps zmm6, zmm5, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x56,0x48,0x52,0x71,0x7f]
          vdpbf16ps zmm6, zmm5, zmmword ptr [ecx + 8128]

// CHECK: vdpbf16ps zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf2,0x56,0xdf,0x52,0x72,0x80]
          vdpbf16ps zmm6 {k7} {z}, zmm5, dword ptr [edx - 512]{1to16}

