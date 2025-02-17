// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvtbf162ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x69,0xf7]
          vcvtbf162ibs xmm22, xmm23

// CHECK: vcvtbf162ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x69,0xf7]
          vcvtbf162ibs xmm22 {k7}, xmm23

// CHECK: vcvtbf162ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x69,0xf7]
          vcvtbf162ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvtbf162ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x69,0xf7]
          vcvtbf162ibs zmm22, zmm23

// CHECK: vcvtbf162ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x69,0xf7]
          vcvtbf162ibs zmm22 {k7}, zmm23

// CHECK: vcvtbf162ibs zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x69,0xf7]
          vcvtbf162ibs zmm22 {k7} {z}, zmm23

// CHECK: vcvtbf162ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x69,0xf7]
          vcvtbf162ibs ymm22, ymm23

// CHECK: vcvtbf162ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x69,0xf7]
          vcvtbf162ibs ymm22 {k7}, ymm23

// CHECK: vcvtbf162ibs ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x69,0xf7]
          vcvtbf162ibs ymm22 {k7} {z}, ymm23

// CHECK: vcvtbf162ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162ibs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x18,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162ibs xmm22, word ptr [rip]{1to8}

// CHECK: vcvtbf162ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x69,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbf162ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x69,0x71,0x7f]
          vcvtbf162ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtbf162ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x9f,0x69,0x72,0x80]
          vcvtbf162ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtbf162ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162ibs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0x38,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162ibs ymm22, word ptr [rip]{1to16}

// CHECK: vcvtbf162ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x69,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbf162ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x69,0x71,0x7f]
          vcvtbf162ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtbf162ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0xbf,0x69,0x72,0x80]
          vcvtbf162ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbf162ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162ibs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0x58,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162ibs zmm22, word ptr [rip]{1to32}

// CHECK: vcvtbf162ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x69,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbf162ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x69,0x71,0x7f]
          vcvtbf162ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtbf162ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0xdf,0x69,0x72,0x80]
          vcvtbf162ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtbf162iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x6b,0xf7]
          vcvtbf162iubs xmm22, xmm23

// CHECK: vcvtbf162iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x6b,0xf7]
          vcvtbf162iubs xmm22 {k7}, xmm23

// CHECK: vcvtbf162iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x6b,0xf7]
          vcvtbf162iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvtbf162iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x6b,0xf7]
          vcvtbf162iubs zmm22, zmm23

// CHECK: vcvtbf162iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x6b,0xf7]
          vcvtbf162iubs zmm22 {k7}, zmm23

// CHECK: vcvtbf162iubs zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x6b,0xf7]
          vcvtbf162iubs zmm22 {k7} {z}, zmm23

// CHECK: vcvtbf162iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x6b,0xf7]
          vcvtbf162iubs ymm22, ymm23

// CHECK: vcvtbf162iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x6b,0xf7]
          vcvtbf162iubs ymm22 {k7}, ymm23

// CHECK: vcvtbf162iubs ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x6b,0xf7]
          vcvtbf162iubs ymm22 {k7} {z}, ymm23

// CHECK: vcvtbf162iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162iubs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x18,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162iubs xmm22, word ptr [rip]{1to8}

// CHECK: vcvtbf162iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x6b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtbf162iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x6b,0x71,0x7f]
          vcvtbf162iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtbf162iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x9f,0x6b,0x72,0x80]
          vcvtbf162iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtbf162iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162iubs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0x38,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162iubs ymm22, word ptr [rip]{1to16}

// CHECK: vcvtbf162iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x6b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtbf162iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x6b,0x71,0x7f]
          vcvtbf162iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtbf162iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0xbf,0x6b,0x72,0x80]
          vcvtbf162iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtbf162iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtbf162iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtbf162iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtbf162iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtbf162iubs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0x58,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtbf162iubs zmm22, word ptr [rip]{1to32}

// CHECK: vcvtbf162iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x6b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtbf162iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x6b,0x71,0x7f]
          vcvtbf162iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtbf162iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0xdf,0x6b,0x72,0x80]
          vcvtbf162iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtph2ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x69,0xf7]
          vcvtph2ibs xmm22, xmm23

// CHECK: vcvtph2ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x69,0xf7]
          vcvtph2ibs xmm22 {k7}, xmm23

// CHECK: vcvtph2ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x69,0xf7]
          vcvtph2ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvtph2ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x69,0xf7]
          vcvtph2ibs zmm22, zmm23

// CHECK: vcvtph2ibs zmm22, zmm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x69,0xf7]
          vcvtph2ibs zmm22, zmm23, {rn-sae}

// CHECK: vcvtph2ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x69,0xf7]
          vcvtph2ibs zmm22 {k7}, zmm23

// CHECK: vcvtph2ibs zmm22 {k7} {z}, zmm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0xff,0x69,0xf7]
          vcvtph2ibs zmm22 {k7} {z}, zmm23, {rz-sae}

// CHECK: vcvtph2ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x69,0xf7]
          vcvtph2ibs ymm22, ymm23

// CHECK: vcvtph2ibs ymm22, ymm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x69,0xf7]
          vcvtph2ibs ymm22, ymm23, {rn-sae}

// CHECK: vcvtph2ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x69,0xf7]
          vcvtph2ibs ymm22 {k7}, ymm23

// CHECK: vcvtph2ibs ymm22 {k7} {z}, ymm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x69,0xf7]
          vcvtph2ibs ymm22 {k7} {z}, ymm23, {rz-sae}

// CHECK: vcvtph2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2ibs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtph2ibs xmm22, word ptr [rip]{1to8}

// CHECK: vcvtph2ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x69,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtph2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x69,0x71,0x7f]
          vcvtph2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x69,0x72,0x80]
          vcvtph2ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2ibs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtph2ibs ymm22, word ptr [rip]{1to16}

// CHECK: vcvtph2ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x69,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtph2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x69,0x71,0x7f]
          vcvtph2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtph2ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x69,0x72,0x80]
          vcvtph2ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtph2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2ibs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtph2ibs zmm22, word ptr [rip]{1to32}

// CHECK: vcvtph2ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x69,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtph2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x69,0x71,0x7f]
          vcvtph2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtph2ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x69,0x72,0x80]
          vcvtph2ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtph2iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6b,0xf7]
          vcvtph2iubs xmm22, xmm23

// CHECK: vcvtph2iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x6b,0xf7]
          vcvtph2iubs xmm22 {k7}, xmm23

// CHECK: vcvtph2iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x6b,0xf7]
          vcvtph2iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvtph2iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6b,0xf7]
          vcvtph2iubs zmm22, zmm23

// CHECK: vcvtph2iubs zmm22, zmm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x6b,0xf7]
          vcvtph2iubs zmm22, zmm23, {rn-sae}

// CHECK: vcvtph2iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x6b,0xf7]
          vcvtph2iubs zmm22 {k7}, zmm23

// CHECK: vcvtph2iubs zmm22 {k7} {z}, zmm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0xff,0x6b,0xf7]
          vcvtph2iubs zmm22 {k7} {z}, zmm23, {rz-sae}

// CHECK: vcvtph2iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6b,0xf7]
          vcvtph2iubs ymm22, ymm23

// CHECK: vcvtph2iubs ymm22, ymm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x6b,0xf7]
          vcvtph2iubs ymm22, ymm23, {rn-sae}

// CHECK: vcvtph2iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x6b,0xf7]
          vcvtph2iubs ymm22 {k7}, ymm23

// CHECK: vcvtph2iubs ymm22 {k7} {z}, ymm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x78,0xff,0x6b,0xf7]
          vcvtph2iubs ymm22 {k7} {z}, ymm23, {rz-sae}

// CHECK: vcvtph2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2iubs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2iubs xmm22, word ptr [rip]{1to8}

// CHECK: vcvtph2iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x6b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtph2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x6b,0x71,0x7f]
          vcvtph2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x6b,0x72,0x80]
          vcvtph2iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2iubs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2iubs ymm22, word ptr [rip]{1to16}

// CHECK: vcvtph2iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x6b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtph2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x6b,0x71,0x7f]
          vcvtph2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtph2iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x6b,0x72,0x80]
          vcvtph2iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtph2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtph2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtph2iubs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtph2iubs zmm22, word ptr [rip]{1to32}

// CHECK: vcvtph2iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x6b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtph2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x6b,0x71,0x7f]
          vcvtph2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtph2iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x6b,0x72,0x80]
          vcvtph2iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvtps2ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x69,0xf7]
          vcvtps2ibs xmm22, xmm23

// CHECK: vcvtps2ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x69,0xf7]
          vcvtps2ibs xmm22 {k7}, xmm23

// CHECK: vcvtps2ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x69,0xf7]
          vcvtps2ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvtps2ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x69,0xf7]
          vcvtps2ibs zmm22, zmm23

// CHECK: vcvtps2ibs zmm22, zmm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x69,0xf7]
          vcvtps2ibs zmm22, zmm23, {rn-sae}

// CHECK: vcvtps2ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x69,0xf7]
          vcvtps2ibs zmm22 {k7}, zmm23

// CHECK: vcvtps2ibs zmm22 {k7} {z}, zmm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0xff,0x69,0xf7]
          vcvtps2ibs zmm22 {k7} {z}, zmm23, {rz-sae}

// CHECK: vcvtps2ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x69,0xf7]
          vcvtps2ibs ymm22, ymm23

// CHECK: vcvtps2ibs ymm22, ymm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x69,0xf7]
          vcvtps2ibs ymm22, ymm23, {rn-sae}

// CHECK: vcvtps2ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x69,0xf7]
          vcvtps2ibs ymm22 {k7}, ymm23

// CHECK: vcvtps2ibs ymm22 {k7} {z}, ymm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x69,0xf7]
          vcvtps2ibs ymm22 {k7} {z}, ymm23, {rz-sae}

// CHECK: vcvtps2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2ibs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtps2ibs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvtps2ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x69,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtps2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x69,0x71,0x7f]
          vcvtps2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtps2ibs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x69,0x72,0x80]
          vcvtps2ibs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtps2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2ibs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtps2ibs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvtps2ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x69,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtps2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x69,0x71,0x7f]
          vcvtps2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtps2ibs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x69,0x72,0x80]
          vcvtps2ibs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtps2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x69,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x69,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2ibs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x69,0x35,0x00,0x00,0x00,0x00]
          vcvtps2ibs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvtps2ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x69,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtps2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x69,0x71,0x7f]
          vcvtps2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtps2ibs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x69,0x72,0x80]
          vcvtps2ibs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vcvtps2iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6b,0xf7]
          vcvtps2iubs xmm22, xmm23

// CHECK: vcvtps2iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x6b,0xf7]
          vcvtps2iubs xmm22 {k7}, xmm23

// CHECK: vcvtps2iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x6b,0xf7]
          vcvtps2iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvtps2iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6b,0xf7]
          vcvtps2iubs zmm22, zmm23

// CHECK: vcvtps2iubs zmm22, zmm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x6b,0xf7]
          vcvtps2iubs zmm22, zmm23, {rn-sae}

// CHECK: vcvtps2iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x6b,0xf7]
          vcvtps2iubs zmm22 {k7}, zmm23

// CHECK: vcvtps2iubs zmm22 {k7} {z}, zmm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0xff,0x6b,0xf7]
          vcvtps2iubs zmm22 {k7} {z}, zmm23, {rz-sae}

// CHECK: vcvtps2iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6b,0xf7]
          vcvtps2iubs ymm22, ymm23

// CHECK: vcvtps2iubs ymm22, ymm23, {rn-sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x6b,0xf7]
          vcvtps2iubs ymm22, ymm23, {rn-sae}

// CHECK: vcvtps2iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x6b,0xf7]
          vcvtps2iubs ymm22 {k7}, ymm23

// CHECK: vcvtps2iubs ymm22 {k7} {z}, ymm23, {rz-sae}
// CHECK: encoding: [0x62,0xa5,0x79,0xff,0x6b,0xf7]
          vcvtps2iubs ymm22 {k7} {z}, ymm23, {rz-sae}

// CHECK: vcvtps2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2iubs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtps2iubs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvtps2iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x6b,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvtps2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x6b,0x71,0x7f]
          vcvtps2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvtps2iubs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x6b,0x72,0x80]
          vcvtps2iubs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtps2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2iubs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtps2iubs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvtps2iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x6b,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvtps2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x6b,0x71,0x7f]
          vcvtps2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvtps2iubs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x6b,0x72,0x80]
          vcvtps2iubs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtps2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x6b,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvtps2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvtps2iubs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x6b,0x35,0x00,0x00,0x00,0x00]
          vcvtps2iubs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvtps2iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x6b,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvtps2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x6b,0x71,0x7f]
          vcvtps2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvtps2iubs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x6b,0x72,0x80]
          vcvtps2iubs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vcvttbf162ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x68,0xf7]
          vcvttbf162ibs xmm22, xmm23

// CHECK: vcvttbf162ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x68,0xf7]
          vcvttbf162ibs xmm22 {k7}, xmm23

// CHECK: vcvttbf162ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x68,0xf7]
          vcvttbf162ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvttbf162ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x68,0xf7]
          vcvttbf162ibs zmm22, zmm23

// CHECK: vcvttbf162ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x68,0xf7]
          vcvttbf162ibs zmm22 {k7}, zmm23

// CHECK: vcvttbf162ibs zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x68,0xf7]
          vcvttbf162ibs zmm22 {k7} {z}, zmm23

// CHECK: vcvttbf162ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x68,0xf7]
          vcvttbf162ibs ymm22, ymm23

// CHECK: vcvttbf162ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x68,0xf7]
          vcvttbf162ibs ymm22 {k7}, ymm23

// CHECK: vcvttbf162ibs ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x68,0xf7]
          vcvttbf162ibs ymm22 {k7} {z}, ymm23

// CHECK: vcvttbf162ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162ibs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x18,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162ibs xmm22, word ptr [rip]{1to8}

// CHECK: vcvttbf162ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttbf162ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x68,0x71,0x7f]
          vcvttbf162ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttbf162ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x9f,0x68,0x72,0x80]
          vcvttbf162ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttbf162ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162ibs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0x38,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162ibs ymm22, word ptr [rip]{1to16}

// CHECK: vcvttbf162ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttbf162ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x68,0x71,0x7f]
          vcvttbf162ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttbf162ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0xbf,0x68,0x72,0x80]
          vcvttbf162ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvttbf162ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162ibs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0x58,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162ibs zmm22, word ptr [rip]{1to32}

// CHECK: vcvttbf162ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x68,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttbf162ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x68,0x71,0x7f]
          vcvttbf162ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttbf162ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0xdf,0x68,0x72,0x80]
          vcvttbf162ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvttbf162iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x6a,0xf7]
          vcvttbf162iubs xmm22, xmm23

// CHECK: vcvttbf162iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x0f,0x6a,0xf7]
          vcvttbf162iubs xmm22 {k7}, xmm23

// CHECK: vcvttbf162iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x8f,0x6a,0xf7]
          vcvttbf162iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvttbf162iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x6a,0xf7]
          vcvttbf162iubs zmm22, zmm23

// CHECK: vcvttbf162iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x4f,0x6a,0xf7]
          vcvttbf162iubs zmm22 {k7}, zmm23

// CHECK: vcvttbf162iubs zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xcf,0x6a,0xf7]
          vcvttbf162iubs zmm22 {k7} {z}, zmm23

// CHECK: vcvttbf162iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x6a,0xf7]
          vcvttbf162iubs ymm22, ymm23

// CHECK: vcvttbf162iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0x2f,0x6a,0xf7]
          vcvttbf162iubs ymm22 {k7}, ymm23

// CHECK: vcvttbf162iubs ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7f,0xaf,0x6a,0xf7]
          vcvttbf162iubs ymm22 {k7} {z}, ymm23

// CHECK: vcvttbf162iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x08,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x0f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162iubs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x18,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162iubs xmm22, word ptr [rip]{1to8}

// CHECK: vcvttbf162iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7f,0x08,0x6a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttbf162iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7f,0x8f,0x6a,0x71,0x7f]
          vcvttbf162iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttbf162iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7f,0x9f,0x6a,0x72,0x80]
          vcvttbf162iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttbf162iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x28,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x2f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162iubs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0x38,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162iubs ymm22, word ptr [rip]{1to16}

// CHECK: vcvttbf162iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7f,0x28,0x6a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttbf162iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7f,0xaf,0x6a,0x71,0x7f]
          vcvttbf162iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttbf162iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7f,0xbf,0x6a,0x72,0x80]
          vcvttbf162iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvttbf162iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7f,0x48,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttbf162iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttbf162iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7f,0x4f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttbf162iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttbf162iubs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0x58,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttbf162iubs zmm22, word ptr [rip]{1to32}

// CHECK: vcvttbf162iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7f,0x48,0x6a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttbf162iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7f,0xcf,0x6a,0x71,0x7f]
          vcvttbf162iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttbf162iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7f,0xdf,0x6a,0x72,0x80]
          vcvttbf162iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvttph2ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x68,0xf7]
          vcvttph2ibs xmm22, xmm23

// CHECK: vcvttph2ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x68,0xf7]
          vcvttph2ibs xmm22 {k7}, xmm23

// CHECK: vcvttph2ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x68,0xf7]
          vcvttph2ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvttph2ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x68,0xf7]
          vcvttph2ibs zmm22, zmm23

// CHECK: vcvttph2ibs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x68,0xf7]
          vcvttph2ibs zmm22, zmm23, {sae}

// CHECK: vcvttph2ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x68,0xf7]
          vcvttph2ibs zmm22 {k7}, zmm23

// CHECK: vcvttph2ibs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x9f,0x68,0xf7]
          vcvttph2ibs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttph2ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x68,0xf7]
          vcvttph2ibs ymm22, ymm23

// CHECK: vcvttph2ibs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x68,0xf7]
          vcvttph2ibs ymm22, ymm23, {sae}

// CHECK: vcvttph2ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x68,0xf7]
          vcvttph2ibs ymm22 {k7}, ymm23

// CHECK: vcvttph2ibs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x68,0xf7]
          vcvttph2ibs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttph2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2ibs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttph2ibs xmm22, word ptr [rip]{1to8}

// CHECK: vcvttph2ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttph2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x68,0x71,0x7f]
          vcvttph2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x68,0x72,0x80]
          vcvttph2ibs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2ibs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttph2ibs ymm22, word ptr [rip]{1to16}

// CHECK: vcvttph2ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttph2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x68,0x71,0x7f]
          vcvttph2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttph2ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x68,0x72,0x80]
          vcvttph2ibs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvttph2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2ibs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttph2ibs zmm22, word ptr [rip]{1to32}

// CHECK: vcvttph2ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x68,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttph2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x68,0x71,0x7f]
          vcvttph2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttph2ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x68,0x72,0x80]
          vcvttph2ibs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvttph2iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6a,0xf7]
          vcvttph2iubs xmm22, xmm23

// CHECK: vcvttph2iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x6a,0xf7]
          vcvttph2iubs xmm22 {k7}, xmm23

// CHECK: vcvttph2iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x6a,0xf7]
          vcvttph2iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvttph2iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6a,0xf7]
          vcvttph2iubs zmm22, zmm23

// CHECK: vcvttph2iubs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x6a,0xf7]
          vcvttph2iubs zmm22, zmm23, {sae}

// CHECK: vcvttph2iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x6a,0xf7]
          vcvttph2iubs zmm22 {k7}, zmm23

// CHECK: vcvttph2iubs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x9f,0x6a,0xf7]
          vcvttph2iubs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttph2iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6a,0xf7]
          vcvttph2iubs ymm22, ymm23

// CHECK: vcvttph2iubs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x6a,0xf7]
          vcvttph2iubs ymm22, ymm23, {sae}

// CHECK: vcvttph2iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x6a,0xf7]
          vcvttph2iubs ymm22 {k7}, ymm23

// CHECK: vcvttph2iubs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x6a,0xf7]
          vcvttph2iubs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttph2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2iubs xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttph2iubs xmm22, word ptr [rip]{1to8}

// CHECK: vcvttph2iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x6a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttph2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x6a,0x71,0x7f]
          vcvttph2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x6a,0x72,0x80]
          vcvttph2iubs xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2iubs ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttph2iubs ymm22, word ptr [rip]{1to16}

// CHECK: vcvttph2iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x6a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttph2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x6a,0x71,0x7f]
          vcvttph2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttph2iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x6a,0x72,0x80]
          vcvttph2iubs ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvttph2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttph2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttph2iubs zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttph2iubs zmm22, word ptr [rip]{1to32}

// CHECK: vcvttph2iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x6a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttph2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x6a,0x71,0x7f]
          vcvttph2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttph2iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x6a,0x72,0x80]
          vcvttph2iubs zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vcvttps2ibs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x68,0xf7]
          vcvttps2ibs xmm22, xmm23

// CHECK: vcvttps2ibs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x68,0xf7]
          vcvttps2ibs xmm22 {k7}, xmm23

// CHECK: vcvttps2ibs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x68,0xf7]
          vcvttps2ibs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2ibs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x68,0xf7]
          vcvttps2ibs zmm22, zmm23

// CHECK: vcvttps2ibs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x68,0xf7]
          vcvttps2ibs zmm22, zmm23, {sae}

// CHECK: vcvttps2ibs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x68,0xf7]
          vcvttps2ibs zmm22 {k7}, zmm23

// CHECK: vcvttps2ibs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x9f,0x68,0xf7]
          vcvttps2ibs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttps2ibs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x68,0xf7]
          vcvttps2ibs ymm22, ymm23

// CHECK: vcvttps2ibs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x68,0xf7]
          vcvttps2ibs ymm22, ymm23, {sae}

// CHECK: vcvttps2ibs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x68,0xf7]
          vcvttps2ibs ymm22 {k7}, ymm23

// CHECK: vcvttps2ibs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x68,0xf7]
          vcvttps2ibs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2ibs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2ibs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2ibs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttps2ibs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2ibs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2ibs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x68,0x71,0x7f]
          vcvttps2ibs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2ibs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x68,0x72,0x80]
          vcvttps2ibs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2ibs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2ibs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2ibs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttps2ibs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2ibs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2ibs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x68,0x71,0x7f]
          vcvttps2ibs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2ibs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x68,0x72,0x80]
          vcvttps2ibs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvttps2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x68,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2ibs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x68,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2ibs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2ibs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x68,0x35,0x00,0x00,0x00,0x00]
          vcvttps2ibs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvttps2ibs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x68,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2ibs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttps2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x68,0x71,0x7f]
          vcvttps2ibs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttps2ibs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x68,0x72,0x80]
          vcvttps2ibs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vcvttps2iubs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6a,0xf7]
          vcvttps2iubs xmm22, xmm23

// CHECK: vcvttps2iubs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x6a,0xf7]
          vcvttps2iubs xmm22 {k7}, xmm23

// CHECK: vcvttps2iubs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x6a,0xf7]
          vcvttps2iubs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2iubs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6a,0xf7]
          vcvttps2iubs zmm22, zmm23

// CHECK: vcvttps2iubs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x6a,0xf7]
          vcvttps2iubs zmm22, zmm23, {sae}

// CHECK: vcvttps2iubs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x6a,0xf7]
          vcvttps2iubs zmm22 {k7}, zmm23

// CHECK: vcvttps2iubs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x9f,0x6a,0xf7]
          vcvttps2iubs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttps2iubs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6a,0xf7]
          vcvttps2iubs ymm22, ymm23

// CHECK: vcvttps2iubs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x6a,0xf7]
          vcvttps2iubs ymm22, ymm23, {sae}

// CHECK: vcvttps2iubs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x6a,0xf7]
          vcvttps2iubs ymm22 {k7}, ymm23

// CHECK: vcvttps2iubs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x6a,0xf7]
          vcvttps2iubs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2iubs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2iubs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2iubs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttps2iubs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2iubs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x6a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2iubs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x6a,0x71,0x7f]
          vcvttps2iubs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2iubs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x6a,0x72,0x80]
          vcvttps2iubs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2iubs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2iubs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2iubs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttps2iubs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2iubs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x6a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2iubs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x6a,0x71,0x7f]
          vcvttps2iubs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2iubs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x6a,0x72,0x80]
          vcvttps2iubs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvttps2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2iubs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x6a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2iubs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2iubs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x6a,0x35,0x00,0x00,0x00,0x00]
          vcvttps2iubs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvttps2iubs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x6a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2iubs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttps2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x6a,0x71,0x7f]
          vcvttps2iubs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttps2iubs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x6a,0x72,0x80]
          vcvttps2iubs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

