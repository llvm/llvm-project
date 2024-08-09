// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvt2ps2phx ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x45,0x20,0x67,0xf0]
          vcvt2ps2phx ymm22, ymm23, ymm24

// CHECK: vcvt2ps2phx ymm22, ymm23, ymm24, {rn-sae}
// CHECK: encoding: [0x62,0x82,0x41,0x10,0x67,0xf0]
          vcvt2ps2phx ymm22, ymm23, ymm24, {rn-sae}

// CHECK: vcvt2ps2phx ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x45,0x27,0x67,0xf0]
          vcvt2ps2phx ymm22 {k7}, ymm23, ymm24

// CHECK: vcvt2ps2phx ymm22 {k7} {z}, ymm23, ymm24, {rz-sae}
// CHECK: encoding: [0x62,0x82,0x41,0xf7,0x67,0xf0]
          vcvt2ps2phx ymm22 {k7} {z}, ymm23, ymm24, {rz-sae}

// CHECK: vcvt2ps2phx zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x45,0x40,0x67,0xf0]
          vcvt2ps2phx zmm22, zmm23, zmm24

// CHECK: vcvt2ps2phx zmm22, zmm23, zmm24, {rn-sae}
// CHECK: encoding: [0x62,0x82,0x45,0x10,0x67,0xf0]
          vcvt2ps2phx zmm22, zmm23, zmm24, {rn-sae}

// CHECK: vcvt2ps2phx zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x45,0x47,0x67,0xf0]
          vcvt2ps2phx zmm22 {k7}, zmm23, zmm24

// CHECK: vcvt2ps2phx zmm22 {k7} {z}, zmm23, zmm24, {rz-sae}
// CHECK: encoding: [0x62,0x82,0x45,0xf7,0x67,0xf0]
          vcvt2ps2phx zmm22 {k7} {z}, zmm23, zmm24, {rz-sae}

// CHECK: vcvt2ps2phx xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x45,0x00,0x67,0xf0]
          vcvt2ps2phx xmm22, xmm23, xmm24

// CHECK: vcvt2ps2phx xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x45,0x07,0x67,0xf0]
          vcvt2ps2phx xmm22 {k7}, xmm23, xmm24

// CHECK: vcvt2ps2phx xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x45,0x87,0x67,0xf0]
          vcvt2ps2phx xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvt2ps2phx zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x45,0x40,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvt2ps2phx zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x45,0x47,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvt2ps2phx zmm22, zmm23, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe2,0x45,0x50,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx zmm22, zmm23, dword ptr [rip]{1to16}

// CHECK: vcvt2ps2phx zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x45,0x40,0x67,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ps2phx zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvt2ps2phx zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x45,0xc7,0x67,0x71,0x7f]
          vcvt2ps2phx zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvt2ps2phx zmm22 {k7} {z}, zmm23, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe2,0x45,0xd7,0x67,0x72,0x80]
          vcvt2ps2phx zmm22 {k7} {z}, zmm23, dword ptr [rdx - 512]{1to16}

// CHECK: vcvt2ps2phx ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x45,0x20,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvt2ps2phx ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x45,0x27,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvt2ps2phx ymm22, ymm23, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe2,0x45,0x30,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx ymm22, ymm23, dword ptr [rip]{1to8}

// CHECK: vcvt2ps2phx ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x45,0x20,0x67,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ps2phx ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvt2ps2phx ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x45,0xa7,0x67,0x71,0x7f]
          vcvt2ps2phx ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvt2ps2phx ymm22 {k7} {z}, ymm23, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe2,0x45,0xb7,0x67,0x72,0x80]
          vcvt2ps2phx ymm22 {k7} {z}, ymm23, dword ptr [rdx - 512]{1to8}

// CHECK: vcvt2ps2phx xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x45,0x00,0x67,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvt2ps2phx xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvt2ps2phx xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x45,0x07,0x67,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvt2ps2phx xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvt2ps2phx xmm22, xmm23, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe2,0x45,0x10,0x67,0x35,0x00,0x00,0x00,0x00]
          vcvt2ps2phx xmm22, xmm23, dword ptr [rip]{1to4}

// CHECK: vcvt2ps2phx xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x45,0x00,0x67,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ps2phx xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvt2ps2phx xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x45,0x87,0x67,0x71,0x7f]
          vcvt2ps2phx xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvt2ps2phx xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe2,0x45,0x97,0x67,0x72,0x80]
          vcvt2ps2phx xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtbiasph2bf8 ymm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x44,0x40,0x74,0xf0]
          vcvtbiasph2bf8 ymm22, zmm23, zmm24

// CHECK: vcvtbiasph2bf8 ymm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x44,0x47,0x74,0xf0]
          vcvtbiasph2bf8 ymm22 {k7}, zmm23, zmm24

// CHECK: vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x44,0xc7,0x74,0xf0]
          vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtbiasph2bf8 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x44,0x00,0x74,0xf0]
          vcvtbiasph2bf8 xmm22, xmm23, xmm24

// CHECK: vcvtbiasph2bf8 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x44,0x07,0x74,0xf0]
          vcvtbiasph2bf8 xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x44,0x87,0x74,0xf0]
          vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtbiasph2bf8 xmm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x44,0x20,0x74,0xf0]
          vcvtbiasph2bf8 xmm22, ymm23, ymm24

// CHECK: vcvtbiasph2bf8 xmm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x44,0x27,0x74,0xf0]
          vcvtbiasph2bf8 xmm22 {k7}, ymm23, ymm24

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x44,0xa7,0x74,0xf0]
          vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtbiasph2bf8 xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x44,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8 xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x44,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8 xmm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe2,0x44,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8 xmm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtbiasph2bf8 xmm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x44,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8 xmm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x44,0xa7,0x74,0x71,0x7f]
          vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe2,0x44,0xb7,0x74,0x72,0x80]
          vcvtbiasph2bf8 xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbiasph2bf8 ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x44,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8 ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x44,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8 ymm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe2,0x44,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8 ymm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtbiasph2bf8 ymm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x44,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8 ymm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x44,0xc7,0x74,0x71,0x7f]
          vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe2,0x44,0xd7,0x74,0x72,0x80]
          vcvtbiasph2bf8 ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtbiasph2bf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x44,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x44,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe2,0x44,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtbiasph2bf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x44,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x44,0x87,0x74,0x71,0x7f]
          vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe2,0x44,0x97,0x74,0x72,0x80]
          vcvtbiasph2bf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtbiasph2bf8s ymm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x74,0xf0]
          vcvtbiasph2bf8s ymm22, zmm23, zmm24

// CHECK: vcvtbiasph2bf8s ymm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x74,0xf0]
          vcvtbiasph2bf8s ymm22 {k7}, zmm23, zmm24

// CHECK: vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x74,0xf0]
          vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtbiasph2bf8s xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x74,0xf0]
          vcvtbiasph2bf8s xmm22, xmm23, xmm24

// CHECK: vcvtbiasph2bf8s xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x74,0xf0]
          vcvtbiasph2bf8s xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x74,0xf0]
          vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtbiasph2bf8s xmm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x74,0xf0]
          vcvtbiasph2bf8s xmm22, ymm23, ymm24

// CHECK: vcvtbiasph2bf8s xmm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x74,0xf0]
          vcvtbiasph2bf8s xmm22 {k7}, ymm23, ymm24

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x74,0xf0]
          vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtbiasph2bf8s xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8s xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8s xmm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s xmm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtbiasph2bf8s xmm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8s xmm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x74,0x71,0x7f]
          vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x74,0x72,0x80]
          vcvtbiasph2bf8s xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbiasph2bf8s ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8s ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8s ymm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s ymm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtbiasph2bf8s ymm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8s ymm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x74,0x71,0x7f]
          vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x74,0x72,0x80]
          vcvtbiasph2bf8s ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtbiasph2bf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2bf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2bf8s xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2bf8s xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtbiasph2bf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x74,0x71,0x7f]
          vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x74,0x72,0x80]
          vcvtbiasph2bf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtbiasph2hf8 ymm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x18,0xf0]
          vcvtbiasph2hf8 ymm22, zmm23, zmm24

// CHECK: vcvtbiasph2hf8 ymm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x18,0xf0]
          vcvtbiasph2hf8 ymm22 {k7}, zmm23, zmm24

// CHECK: vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x18,0xf0]
          vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtbiasph2hf8 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x18,0xf0]
          vcvtbiasph2hf8 xmm22, xmm23, xmm24

// CHECK: vcvtbiasph2hf8 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x18,0xf0]
          vcvtbiasph2hf8 xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x18,0xf0]
          vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtbiasph2hf8 xmm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x18,0xf0]
          vcvtbiasph2hf8 xmm22, ymm23, ymm24

// CHECK: vcvtbiasph2hf8 xmm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x18,0xf0]
          vcvtbiasph2hf8 xmm22 {k7}, ymm23, ymm24

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x18,0xf0]
          vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtbiasph2hf8 xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8 xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8 xmm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8 xmm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtbiasph2hf8 xmm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8 xmm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x18,0x71,0x7f]
          vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x18,0x72,0x80]
          vcvtbiasph2hf8 xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbiasph2hf8 ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8 ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8 ymm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8 ymm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtbiasph2hf8 ymm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8 ymm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x18,0x71,0x7f]
          vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x18,0x72,0x80]
          vcvtbiasph2hf8 ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtbiasph2hf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtbiasph2hf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x18,0x71,0x7f]
          vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x18,0x72,0x80]
          vcvtbiasph2hf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtbiasph2hf8s ymm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x40,0x1b,0xf0]
          vcvtbiasph2hf8s ymm22, zmm23, zmm24

// CHECK: vcvtbiasph2hf8s ymm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0x47,0x1b,0xf0]
          vcvtbiasph2hf8s ymm22 {k7}, zmm23, zmm24

// CHECK: vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x44,0xc7,0x1b,0xf0]
          vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtbiasph2hf8s xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x00,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22, xmm23, xmm24

// CHECK: vcvtbiasph2hf8s xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x07,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x44,0x87,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtbiasph2hf8s xmm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x20,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22, ymm23, ymm24

// CHECK: vcvtbiasph2hf8s xmm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0x27,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22 {k7}, ymm23, ymm24

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x44,0xa7,0x1b,0xf0]
          vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtbiasph2hf8s xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x20,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s xmm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8s xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x27,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s xmm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8s xmm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0x30,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s xmm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtbiasph2hf8s xmm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x44,0x20,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8s xmm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x44,0xa7,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x44,0xb7,0x1b,0x72,0x80]
          vcvtbiasph2hf8s xmm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbiasph2hf8s ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x40,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s ymm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8s ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x47,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s ymm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8s ymm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0x50,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s ymm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtbiasph2hf8s ymm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x44,0x40,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8s ymm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x44,0xc7,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x44,0xd7,0x1b,0x72,0x80]
          vcvtbiasph2hf8s ymm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtbiasph2hf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x44,0x00,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbiasph2hf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x44,0x07,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbiasph2hf8s xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x10,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtbiasph2hf8s xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtbiasph2hf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x44,0x00,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x44,0x87,0x1b,0x71,0x7f]
          vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x44,0x97,0x1b,0x72,0x80]
          vcvtbiasph2hf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvthf82ph xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x1e,0xf7]
          vcvthf82ph xmm22, xmm23

// CHECK: vcvthf82ph xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x1e,0xf7]
          vcvthf82ph xmm22 {k7}, xmm23

// CHECK: vcvthf82ph xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x1e,0xf7]
          vcvthf82ph xmm22 {k7} {z}, xmm23

// CHECK: vcvthf82ph ymm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x1e,0xf7]
          vcvthf82ph ymm22, xmm23

// CHECK: vcvthf82ph ymm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x1e,0xf7]
          vcvthf82ph ymm22 {k7}, xmm23

// CHECK: vcvthf82ph ymm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x1e,0xf7]
          vcvthf82ph ymm22 {k7} {z}, xmm23

// CHECK: vcvthf82ph zmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x1e,0xf7]
          vcvthf82ph zmm22, ymm23

// CHECK: vcvthf82ph zmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x1e,0xf7]
          vcvthf82ph zmm22 {k7}, ymm23

// CHECK: vcvthf82ph zmm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x1e,0xf7]
          vcvthf82ph zmm22 {k7} {z}, ymm23

// CHECK: vcvthf82ph xmm22, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph xmm22, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvthf82ph xmm22 {k7}, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph xmm22 {k7}, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvthf82ph xmm22, qword ptr [rip]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph xmm22, qword ptr [rip]

// CHECK: vcvthf82ph xmm22, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x1e,0x34,0x6d,0x00,0xff,0xff,0xff]
          vcvthf82ph xmm22, qword ptr [2*rbp - 256]

// CHECK: vcvthf82ph xmm22 {k7} {z}, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x1e,0x71,0x7f]
          vcvthf82ph xmm22 {k7} {z}, qword ptr [rcx + 1016]

// CHECK: vcvthf82ph xmm22 {k7} {z}, qword ptr [rdx - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x1e,0x72,0x80]
          vcvthf82ph xmm22 {k7} {z}, qword ptr [rdx - 1024]

// CHECK: vcvthf82ph ymm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph ymm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvthf82ph ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvthf82ph ymm22, xmmword ptr [rip]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph ymm22, xmmword ptr [rip]

// CHECK: vcvthf82ph ymm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x1e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvthf82ph ymm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvthf82ph ymm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x1e,0x71,0x7f]
          vcvthf82ph ymm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvthf82ph ymm22 {k7} {z}, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x1e,0x72,0x80]
          vcvthf82ph ymm22 {k7} {z}, xmmword ptr [rdx - 2048]

// CHECK: vcvthf82ph zmm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x1e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvthf82ph zmm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvthf82ph zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x1e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvthf82ph zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvthf82ph zmm22, ymmword ptr [rip]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x1e,0x35,0x00,0x00,0x00,0x00]
          vcvthf82ph zmm22, ymmword ptr [rip]

// CHECK: vcvthf82ph zmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x1e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvthf82ph zmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvthf82ph zmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x1e,0x71,0x7f]
          vcvthf82ph zmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvthf82ph zmm22 {k7} {z}, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x1e,0x72,0x80]
          vcvthf82ph zmm22 {k7} {z}, ymmword ptr [rdx - 4096]

// CHECK: vcvtne2ph2bf8 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x47,0x20,0x74,0xf0]
          vcvtne2ph2bf8 ymm22, ymm23, ymm24

// CHECK: vcvtne2ph2bf8 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x47,0x27,0x74,0xf0]
          vcvtne2ph2bf8 ymm22 {k7}, ymm23, ymm24

// CHECK: vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x82,0x47,0xa7,0x74,0xf0]
          vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtne2ph2bf8 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x47,0x40,0x74,0xf0]
          vcvtne2ph2bf8 zmm22, zmm23, zmm24

// CHECK: vcvtne2ph2bf8 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x47,0x47,0x74,0xf0]
          vcvtne2ph2bf8 zmm22 {k7}, zmm23, zmm24

// CHECK: vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x82,0x47,0xc7,0x74,0xf0]
          vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtne2ph2bf8 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x47,0x00,0x74,0xf0]
          vcvtne2ph2bf8 xmm22, xmm23, xmm24

// CHECK: vcvtne2ph2bf8 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x47,0x07,0x74,0xf0]
          vcvtne2ph2bf8 xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x82,0x47,0x87,0x74,0xf0]
          vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtne2ph2bf8 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe2,0x47,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtne2ph2bf8 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x47,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtne2ph2bf8 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x47,0xc7,0x74,0x71,0x7f]
          vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe2,0x47,0xd7,0x74,0x72,0x80]
          vcvtne2ph2bf8 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtne2ph2bf8 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe2,0x47,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtne2ph2bf8 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x47,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtne2ph2bf8 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x47,0xa7,0x74,0x71,0x7f]
          vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe2,0x47,0xb7,0x74,0x72,0x80]
          vcvtne2ph2bf8 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtne2ph2bf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x47,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x47,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe2,0x47,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtne2ph2bf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x47,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtne2ph2bf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x47,0x87,0x74,0x71,0x7f]
          vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe2,0x47,0x97,0x74,0x72,0x80]
          vcvtne2ph2bf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtne2ph2bf8s ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x74,0xf0]
          vcvtne2ph2bf8s ymm22, ymm23, ymm24

// CHECK: vcvtne2ph2bf8s ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x74,0xf0]
          vcvtne2ph2bf8s ymm22 {k7}, ymm23, ymm24

// CHECK: vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x74,0xf0]
          vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtne2ph2bf8s zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x74,0xf0]
          vcvtne2ph2bf8s zmm22, zmm23, zmm24

// CHECK: vcvtne2ph2bf8s zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x74,0xf0]
          vcvtne2ph2bf8s zmm22 {k7}, zmm23, zmm24

// CHECK: vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x74,0xf0]
          vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtne2ph2bf8s xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x74,0xf0]
          vcvtne2ph2bf8s xmm22, xmm23, xmm24

// CHECK: vcvtne2ph2bf8s xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x74,0xf0]
          vcvtne2ph2bf8s xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x74,0xf0]
          vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtne2ph2bf8s zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8s zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8s zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8s zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8s zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8s zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtne2ph2bf8s zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtne2ph2bf8s zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x74,0x71,0x7f]
          vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x74,0x72,0x80]
          vcvtne2ph2bf8s zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtne2ph2bf8s ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8s ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8s ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8s ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8s ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8s ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtne2ph2bf8s ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtne2ph2bf8s ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x74,0x71,0x7f]
          vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x74,0x72,0x80]
          vcvtne2ph2bf8s ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtne2ph2bf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2bf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2bf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2bf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2bf8s xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2bf8s xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtne2ph2bf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtne2ph2bf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x74,0x71,0x7f]
          vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x74,0x72,0x80]
          vcvtne2ph2bf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtne2ph2hf8 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x18,0xf0]
          vcvtne2ph2hf8 ymm22, ymm23, ymm24

// CHECK: vcvtne2ph2hf8 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x18,0xf0]
          vcvtne2ph2hf8 ymm22 {k7}, ymm23, ymm24

// CHECK: vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x18,0xf0]
          vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtne2ph2hf8 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x18,0xf0]
          vcvtne2ph2hf8 zmm22, zmm23, zmm24

// CHECK: vcvtne2ph2hf8 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x18,0xf0]
          vcvtne2ph2hf8 zmm22 {k7}, zmm23, zmm24

// CHECK: vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x18,0xf0]
          vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtne2ph2hf8 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x18,0xf0]
          vcvtne2ph2hf8 xmm22, xmm23, xmm24

// CHECK: vcvtne2ph2hf8 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x18,0xf0]
          vcvtne2ph2hf8 xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x18,0xf0]
          vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtne2ph2hf8 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtne2ph2hf8 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtne2ph2hf8 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x18,0x71,0x7f]
          vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x18,0x72,0x80]
          vcvtne2ph2hf8 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtne2ph2hf8 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtne2ph2hf8 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtne2ph2hf8 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x18,0x71,0x7f]
          vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x18,0x72,0x80]
          vcvtne2ph2hf8 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtne2ph2hf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtne2ph2hf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtne2ph2hf8 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x18,0x71,0x7f]
          vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x18,0x72,0x80]
          vcvtne2ph2hf8 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtne2ph2hf8s ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x20,0x1b,0xf0]
          vcvtne2ph2hf8s ymm22, ymm23, ymm24

// CHECK: vcvtne2ph2hf8s ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0x27,0x1b,0xf0]
          vcvtne2ph2hf8s ymm22 {k7}, ymm23, ymm24

// CHECK: vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x47,0xa7,0x1b,0xf0]
          vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vcvtne2ph2hf8s zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x40,0x1b,0xf0]
          vcvtne2ph2hf8s zmm22, zmm23, zmm24

// CHECK: vcvtne2ph2hf8s zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0x47,0x1b,0xf0]
          vcvtne2ph2hf8s zmm22 {k7}, zmm23, zmm24

// CHECK: vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x47,0xc7,0x1b,0xf0]
          vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vcvtne2ph2hf8s xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x00,0x1b,0xf0]
          vcvtne2ph2hf8s xmm22, xmm23, xmm24

// CHECK: vcvtne2ph2hf8s xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x07,0x1b,0xf0]
          vcvtne2ph2hf8s xmm22 {k7}, xmm23, xmm24

// CHECK: vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x47,0x87,0x1b,0xf0]
          vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vcvtne2ph2hf8s zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x40,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8s zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8s zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x47,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8s zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8s zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0x50,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8s zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vcvtne2ph2hf8s zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x47,0x40,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtne2ph2hf8s zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x47,0xc7,0x1b,0x71,0x7f]
          vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x47,0xd7,0x1b,0x72,0x80]
          vcvtne2ph2hf8s zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vcvtne2ph2hf8s ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x20,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8s ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8s ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x27,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8s ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8s ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0x30,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8s ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vcvtne2ph2hf8s ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x47,0x20,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtne2ph2hf8s ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x47,0xa7,0x1b,0x71,0x7f]
          vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x47,0xb7,0x1b,0x72,0x80]
          vcvtne2ph2hf8s ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vcvtne2ph2hf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x47,0x00,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtne2ph2hf8s xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtne2ph2hf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x47,0x07,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtne2ph2hf8s xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtne2ph2hf8s xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x10,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtne2ph2hf8s xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vcvtne2ph2hf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x47,0x00,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtne2ph2hf8s xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x47,0x87,0x1b,0x71,0x7f]
          vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x47,0x97,0x1b,0x72,0x80]
          vcvtne2ph2hf8s xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcvtneph2bf8 xmm22, xmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x08,0x74,0xf7]
          vcvtneph2bf8 xmm22, xmm23

// CHECK: vcvtneph2bf8 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x0f,0x74,0xf7]
          vcvtneph2bf8 xmm22 {k7}, xmm23

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x8f,0x74,0xf7]
          vcvtneph2bf8 xmm22 {k7} {z}, xmm23

// CHECK: vcvtneph2bf8 ymm22, zmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x48,0x74,0xf7]
          vcvtneph2bf8 ymm22, zmm23

// CHECK: vcvtneph2bf8 ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x4f,0x74,0xf7]
          vcvtneph2bf8 ymm22 {k7}, zmm23

// CHECK: vcvtneph2bf8 ymm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa2,0x7e,0xcf,0x74,0xf7]
          vcvtneph2bf8 ymm22 {k7} {z}, zmm23

// CHECK: vcvtneph2bf8 xmm22, ymm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x28,0x74,0xf7]
          vcvtneph2bf8 xmm22, ymm23

// CHECK: vcvtneph2bf8 xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa2,0x7e,0x2f,0x74,0xf7]
          vcvtneph2bf8 xmm22 {k7}, ymm23

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa2,0x7e,0xaf,0x74,0xf7]
          vcvtneph2bf8 xmm22 {k7} {z}, ymm23

// CHECK: vcvtneph2bf8 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x7e,0x08,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2bf8 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2bf8 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x7e,0x0f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2bf8 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2bf8 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe2,0x7e,0x18,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8 xmm22, word ptr [rip]{1to8}

// CHECK: vcvtneph2bf8 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe2,0x7e,0x08,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtneph2bf8 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe2,0x7e,0x8f,0x74,0x71,0x7f]
          vcvtneph2bf8 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe2,0x7e,0x9f,0x74,0x72,0x80]
          vcvtneph2bf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtneph2bf8 xmm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe2,0x7e,0x38,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8 xmm22, word ptr [rip]{1to16}

// CHECK: vcvtneph2bf8 xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe2,0x7e,0x28,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtneph2bf8 xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe2,0x7e,0xaf,0x74,0x71,0x7f]
          vcvtneph2bf8 xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtneph2bf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe2,0x7e,0xbf,0x74,0x72,0x80]
          vcvtneph2bf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtneph2bf8 ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa2,0x7e,0x48,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2bf8 ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2bf8 ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc2,0x7e,0x4f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2bf8 ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2bf8 ymm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe2,0x7e,0x58,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8 ymm22, word ptr [rip]{1to32}

// CHECK: vcvtneph2bf8 ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe2,0x7e,0x48,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtneph2bf8 ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtneph2bf8 ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe2,0x7e,0xcf,0x74,0x71,0x7f]
          vcvtneph2bf8 ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtneph2bf8 ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe2,0x7e,0xdf,0x74,0x72,0x80]
          vcvtneph2bf8 ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtneph2bf8s xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x74,0xf7]
          vcvtneph2bf8s xmm22, xmm23

// CHECK: vcvtneph2bf8s xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x74,0xf7]
          vcvtneph2bf8s xmm22 {k7}, xmm23

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x74,0xf7]
          vcvtneph2bf8s xmm22 {k7} {z}, xmm23

// CHECK: vcvtneph2bf8s ymm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x74,0xf7]
          vcvtneph2bf8s ymm22, zmm23

// CHECK: vcvtneph2bf8s ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x74,0xf7]
          vcvtneph2bf8s ymm22 {k7}, zmm23

// CHECK: vcvtneph2bf8s ymm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x74,0xf7]
          vcvtneph2bf8s ymm22 {k7} {z}, zmm23

// CHECK: vcvtneph2bf8s xmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x74,0xf7]
          vcvtneph2bf8s xmm22, ymm23

// CHECK: vcvtneph2bf8s xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x74,0xf7]
          vcvtneph2bf8s xmm22 {k7}, ymm23

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x74,0xf7]
          vcvtneph2bf8s xmm22 {k7} {z}, ymm23

// CHECK: vcvtneph2bf8s xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2bf8s xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2bf8s xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2bf8s xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2bf8s xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8s xmm22, word ptr [rip]{1to8}

// CHECK: vcvtneph2bf8s xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x74,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtneph2bf8s xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x74,0x71,0x7f]
          vcvtneph2bf8s xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x74,0x72,0x80]
          vcvtneph2bf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtneph2bf8s xmm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8s xmm22, word ptr [rip]{1to16}

// CHECK: vcvtneph2bf8s xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x74,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtneph2bf8s xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x74,0x71,0x7f]
          vcvtneph2bf8s xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtneph2bf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x74,0x72,0x80]
          vcvtneph2bf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtneph2bf8s ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x74,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2bf8s ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2bf8s ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x74,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2bf8s ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2bf8s ymm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x74,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2bf8s ymm22, word ptr [rip]{1to32}

// CHECK: vcvtneph2bf8s ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x74,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtneph2bf8s ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtneph2bf8s ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x74,0x71,0x7f]
          vcvtneph2bf8s ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtneph2bf8s ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x74,0x72,0x80]
          vcvtneph2bf8s ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtneph2hf8 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x18,0xf7]
          vcvtneph2hf8 xmm22, xmm23

// CHECK: vcvtneph2hf8 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x18,0xf7]
          vcvtneph2hf8 xmm22 {k7}, xmm23

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x18,0xf7]
          vcvtneph2hf8 xmm22 {k7} {z}, xmm23

// CHECK: vcvtneph2hf8 ymm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x18,0xf7]
          vcvtneph2hf8 ymm22, zmm23

// CHECK: vcvtneph2hf8 ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x18,0xf7]
          vcvtneph2hf8 ymm22 {k7}, zmm23

// CHECK: vcvtneph2hf8 ymm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x18,0xf7]
          vcvtneph2hf8 ymm22 {k7} {z}, zmm23

// CHECK: vcvtneph2hf8 xmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x18,0xf7]
          vcvtneph2hf8 xmm22, ymm23

// CHECK: vcvtneph2hf8 xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x18,0xf7]
          vcvtneph2hf8 xmm22 {k7}, ymm23

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x18,0xf7]
          vcvtneph2hf8 xmm22 {k7} {z}, ymm23

// CHECK: vcvtneph2hf8 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2hf8 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2hf8 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2hf8 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2hf8 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8 xmm22, word ptr [rip]{1to8}

// CHECK: vcvtneph2hf8 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x18,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtneph2hf8 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x18,0x71,0x7f]
          vcvtneph2hf8 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x18,0x72,0x80]
          vcvtneph2hf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtneph2hf8 xmm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8 xmm22, word ptr [rip]{1to16}

// CHECK: vcvtneph2hf8 xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x18,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtneph2hf8 xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x18,0x71,0x7f]
          vcvtneph2hf8 xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtneph2hf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x18,0x72,0x80]
          vcvtneph2hf8 xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtneph2hf8 ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x18,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2hf8 ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2hf8 ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x18,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2hf8 ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2hf8 ymm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x18,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8 ymm22, word ptr [rip]{1to32}

// CHECK: vcvtneph2hf8 ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x18,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtneph2hf8 ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtneph2hf8 ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x18,0x71,0x7f]
          vcvtneph2hf8 ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtneph2hf8 ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x18,0x72,0x80]
          vcvtneph2hf8 ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtneph2hf8s xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x1b,0xf7]
          vcvtneph2hf8s xmm22, xmm23

// CHECK: vcvtneph2hf8s xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x0f,0x1b,0xf7]
          vcvtneph2hf8s xmm22 {k7}, xmm23

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x8f,0x1b,0xf7]
          vcvtneph2hf8s xmm22 {k7} {z}, xmm23

// CHECK: vcvtneph2hf8s ymm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x1b,0xf7]
          vcvtneph2hf8s ymm22, zmm23

// CHECK: vcvtneph2hf8s ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x4f,0x1b,0xf7]
          vcvtneph2hf8s ymm22 {k7}, zmm23

// CHECK: vcvtneph2hf8s ymm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xcf,0x1b,0xf7]
          vcvtneph2hf8s ymm22 {k7} {z}, zmm23

// CHECK: vcvtneph2hf8s xmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x28,0x1b,0xf7]
          vcvtneph2hf8s xmm22, ymm23

// CHECK: vcvtneph2hf8s xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0x2f,0x1b,0xf7]
          vcvtneph2hf8s xmm22 {k7}, ymm23

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7e,0xaf,0x1b,0xf7]
          vcvtneph2hf8s xmm22 {k7} {z}, ymm23

// CHECK: vcvtneph2hf8s xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x08,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2hf8s xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2hf8s xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x0f,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2hf8s xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2hf8s xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x18,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8s xmm22, word ptr [rip]{1to8}

// CHECK: vcvtneph2hf8s xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7e,0x08,0x1b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtneph2hf8s xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7e,0x8f,0x1b,0x71,0x7f]
          vcvtneph2hf8s xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7e,0x9f,0x1b,0x72,0x80]
          vcvtneph2hf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtneph2hf8s xmm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0x38,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8s xmm22, word ptr [rip]{1to16}

// CHECK: vcvtneph2hf8s xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7e,0x28,0x1b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtneph2hf8s xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7e,0xaf,0x1b,0x71,0x7f]
          vcvtneph2hf8s xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtneph2hf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7e,0xbf,0x1b,0x72,0x80]
          vcvtneph2hf8s xmm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtneph2hf8s ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7e,0x48,0x1b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtneph2hf8s ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtneph2hf8s ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7e,0x4f,0x1b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtneph2hf8s ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtneph2hf8s ymm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0x58,0x1b,0x35,0x00,0x00,0x00,0x00]
          vcvtneph2hf8s ymm22, word ptr [rip]{1to32}

// CHECK: vcvtneph2hf8s ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7e,0x48,0x1b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtneph2hf8s ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtneph2hf8s ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7e,0xcf,0x1b,0x71,0x7f]
          vcvtneph2hf8s ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtneph2hf8s ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7e,0xdf,0x1b,0x72,0x80]
          vcvtneph2hf8s ymm22 {k7} {z}, word ptr [rdx - 256]{1to32}

