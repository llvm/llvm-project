// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vminmaxnepbf16 xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x00,0x52,0xf0,0x7b]
          vminmaxnepbf16 xmm22, xmm23, xmm24, 123

// CHECK: vminmaxnepbf16 xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x07,0x52,0xf0,0x7b]
          vminmaxnepbf16 xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxnepbf16 xmm22 {k7} {z}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x87,0x52,0xf0,0x7b]
          vminmaxnepbf16 xmm22 {k7} {z}, xmm23, xmm24, 123

// CHECK: vminmaxnepbf16 zmm22, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x40,0x52,0xf0,0x7b]
          vminmaxnepbf16 zmm22, zmm23, zmm24, 123

// CHECK: vminmaxnepbf16 zmm22 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x47,0x52,0xf0,0x7b]
          vminmaxnepbf16 zmm22 {k7}, zmm23, zmm24, 123

// CHECK: vminmaxnepbf16 zmm22 {k7} {z}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0xc7,0x52,0xf0,0x7b]
          vminmaxnepbf16 zmm22 {k7} {z}, zmm23, zmm24, 123

// CHECK: vminmaxnepbf16 ymm22, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x20,0x52,0xf0,0x7b]
          vminmaxnepbf16 ymm22, ymm23, ymm24, 123

// CHECK: vminmaxnepbf16 ymm22 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0x27,0x52,0xf0,0x7b]
          vminmaxnepbf16 ymm22 {k7}, ymm23, ymm24, 123

// CHECK: vminmaxnepbf16 ymm22 {k7} {z}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x47,0xa7,0x52,0xf0,0x7b]
          vminmaxnepbf16 ymm22 {k7} {z}, ymm23, ymm24, 123

// CHECK: vminmaxnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x47,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x47,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxnepbf16 ymm22, ymm23, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16 ymm22, ymm23, word ptr [rip]{1to16}, 123

// CHECK: vminmaxnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x47,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vminmaxnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x47,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vminmaxnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}, 123

// CHECK: vminmaxnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x47,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x47,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxnepbf16 xmm22, xmm23, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16 xmm22, xmm23, word ptr [rip]{1to8}, 123

// CHECK: vminmaxnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x47,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vminmaxnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x47,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vminmaxnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0x97,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}, 123

// CHECK: vminmaxnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x47,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x47,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxnepbf16 zmm22, zmm23, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxnepbf16 zmm22, zmm23, word ptr [rip]{1to32}, 123

// CHECK: vminmaxnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x47,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vminmaxnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x47,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vminmaxnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x47,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}, 123

// CHECK: vminmaxpd xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x00,0x52,0xf0,0x7b]
          vminmaxpd xmm22, xmm23, xmm24, 123

// CHECK: vminmaxpd xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x07,0x52,0xf0,0x7b]
          vminmaxpd xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxpd xmm22 {k7} {z}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x87,0x52,0xf0,0x7b]
          vminmaxpd xmm22 {k7} {z}, xmm23, xmm24, 123

// CHECK: vminmaxpd zmm22, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x40,0x52,0xf0,0x7b]
          vminmaxpd zmm22, zmm23, zmm24, 123

// CHECK: vminmaxpd zmm22, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x10,0x52,0xf0,0x7b]
          vminmaxpd zmm22, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxpd zmm22 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x47,0x52,0xf0,0x7b]
          vminmaxpd zmm22 {k7}, zmm23, zmm24, 123

// CHECK: vminmaxpd zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x97,0x52,0xf0,0x7b]
          vminmaxpd zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxpd ymm22, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x20,0x52,0xf0,0x7b]
          vminmaxpd ymm22, ymm23, ymm24, 123

// CHECK: vminmaxpd ymm22, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc1,0x10,0x52,0xf0,0x7b]
          vminmaxpd ymm22, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxpd ymm22 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x27,0x52,0xf0,0x7b]
          vminmaxpd ymm22 {k7}, ymm23, ymm24, 123

// CHECK: vminmaxpd ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc1,0x97,0x52,0xf0,0x7b]
          vminmaxpd ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxpd ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0xc5,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxpd ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0xc5,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxpd ymm22, ymm23, qword ptr [rip]{1to4}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd ymm22, ymm23, qword ptr [rip]{1to4}, 123

// CHECK: vminmaxpd ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxpd ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vminmaxpd ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxpd ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vminmaxpd ymm22 {k7} {z}, ymm23, qword ptr [rdx - 1024]{1to4}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxpd ymm22 {k7} {z}, ymm23, qword ptr [rdx - 1024]{1to4}, 123

// CHECK: vminmaxpd xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0xc5,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxpd xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0xc5,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxpd xmm22, xmm23, qword ptr [rip]{1to2}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd xmm22, xmm23, qword ptr [rip]{1to2}, 123

// CHECK: vminmaxpd xmm22, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxpd xmm22, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vminmaxpd xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxpd xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vminmaxpd xmm22 {k7} {z}, xmm23, qword ptr [rdx - 1024]{1to2}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x97,0x52,0x72,0x80,0x7b]
          vminmaxpd xmm22 {k7} {z}, xmm23, qword ptr [rdx - 1024]{1to2}, 123

// CHECK: vminmaxpd zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0xc5,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxpd zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0xc5,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxpd zmm22, zmm23, qword ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxpd zmm22, zmm23, qword ptr [rip]{1to8}, 123

// CHECK: vminmaxpd zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxpd zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vminmaxpd zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxpd zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vminmaxpd zmm22 {k7} {z}, zmm23, qword ptr [rdx - 1024]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0xc5,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxpd zmm22 {k7} {z}, zmm23, qword ptr [rdx - 1024]{1to8}, 123

// CHECK: vminmaxph xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x00,0x52,0xf0,0x7b]
          vminmaxph xmm22, xmm23, xmm24, 123

// CHECK: vminmaxph xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x07,0x52,0xf0,0x7b]
          vminmaxph xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxph xmm22 {k7} {z}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x87,0x52,0xf0,0x7b]
          vminmaxph xmm22 {k7} {z}, xmm23, xmm24, 123

// CHECK: vminmaxph zmm22, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x40,0x52,0xf0,0x7b]
          vminmaxph zmm22, zmm23, zmm24, 123

// CHECK: vminmaxph zmm22, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x44,0x10,0x52,0xf0,0x7b]
          vminmaxph zmm22, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxph zmm22 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x47,0x52,0xf0,0x7b]
          vminmaxph zmm22 {k7}, zmm23, zmm24, 123

// CHECK: vminmaxph zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x44,0x97,0x52,0xf0,0x7b]
          vminmaxph zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxph ymm22, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x20,0x52,0xf0,0x7b]
          vminmaxph ymm22, ymm23, ymm24, 123

// CHECK: vminmaxph ymm22, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x40,0x10,0x52,0xf0,0x7b]
          vminmaxph ymm22, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxph ymm22 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x27,0x52,0xf0,0x7b]
          vminmaxph ymm22 {k7}, ymm23, ymm24, 123

// CHECK: vminmaxph ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x40,0x97,0x52,0xf0,0x7b]
          vminmaxph ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxph ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x44,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxph ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x44,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxph ymm22, ymm23, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph ymm22, ymm23, word ptr [rip]{1to16}, 123

// CHECK: vminmaxph ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxph ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vminmaxph ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x44,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxph ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vminmaxph ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxph ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}, 123

// CHECK: vminmaxph xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x44,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxph xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x44,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxph xmm22, xmm23, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph xmm22, xmm23, word ptr [rip]{1to8}, 123

// CHECK: vminmaxph xmm22, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxph xmm22, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vminmaxph xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxph xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vminmaxph xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0x97,0x52,0x72,0x80,0x7b]
          vminmaxph xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}, 123

// CHECK: vminmaxph zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x44,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxph zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x44,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxph zmm22, zmm23, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxph zmm22, zmm23, word ptr [rip]{1to32}, 123

// CHECK: vminmaxph zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxph zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vminmaxph zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x44,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxph zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vminmaxph zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x44,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxph zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}, 123

// CHECK: vminmaxps xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x00,0x52,0xf0,0x7b]
          vminmaxps xmm22, xmm23, xmm24, 123

// CHECK: vminmaxps xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x07,0x52,0xf0,0x7b]
          vminmaxps xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxps xmm22 {k7} {z}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x87,0x52,0xf0,0x7b]
          vminmaxps xmm22 {k7} {z}, xmm23, xmm24, 123

// CHECK: vminmaxps zmm22, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x40,0x52,0xf0,0x7b]
          vminmaxps zmm22, zmm23, zmm24, 123

// CHECK: vminmaxps zmm22, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x45,0x10,0x52,0xf0,0x7b]
          vminmaxps zmm22, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxps zmm22 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x47,0x52,0xf0,0x7b]
          vminmaxps zmm22 {k7}, zmm23, zmm24, 123

// CHECK: vminmaxps zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x45,0x97,0x52,0xf0,0x7b]
          vminmaxps zmm22 {k7} {z}, zmm23, zmm24, {sae}, 123

// CHECK: vminmaxps ymm22, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x20,0x52,0xf0,0x7b]
          vminmaxps ymm22, ymm23, ymm24, 123

// CHECK: vminmaxps ymm22, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x41,0x10,0x52,0xf0,0x7b]
          vminmaxps ymm22, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxps ymm22 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x27,0x52,0xf0,0x7b]
          vminmaxps ymm22 {k7}, ymm23, ymm24, 123

// CHECK: vminmaxps ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x41,0x97,0x52,0xf0,0x7b]
          vminmaxps ymm22 {k7} {z}, ymm23, ymm24, {sae}, 123

// CHECK: vminmaxps ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x45,0x20,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxps ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x45,0x27,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxps ymm22, ymm23, dword ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0x30,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps ymm22, ymm23, dword ptr [rip]{1to8}, 123

// CHECK: vminmaxps ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x20,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxps ymm22, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vminmaxps ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x45,0xa7,0x52,0x71,0x7f,0x7b]
          vminmaxps ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vminmaxps ymm22 {k7} {z}, ymm23, dword ptr [rdx - 512]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0xb7,0x52,0x72,0x80,0x7b]
          vminmaxps ymm22 {k7} {z}, ymm23, dword ptr [rdx - 512]{1to8}, 123

// CHECK: vminmaxps xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x45,0x00,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxps xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x45,0x07,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxps xmm22, xmm23, dword ptr [rip]{1to4}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0x10,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps xmm22, xmm23, dword ptr [rip]{1to4}, 123

// CHECK: vminmaxps xmm22, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxps xmm22, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vminmaxps xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x52,0x71,0x7f,0x7b]
          vminmaxps xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vminmaxps xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512]{1to4}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0x97,0x52,0x72,0x80,0x7b]
          vminmaxps xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512]{1to4}, 123

// CHECK: vminmaxps zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x45,0x40,0x52,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxps zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x45,0x47,0x52,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxps zmm22, zmm23, dword ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0x50,0x52,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxps zmm22, zmm23, dword ptr [rip]{1to16}, 123

// CHECK: vminmaxps zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x40,0x52,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxps zmm22, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vminmaxps zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x45,0xc7,0x52,0x71,0x7f,0x7b]
          vminmaxps zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vminmaxps zmm22 {k7} {z}, zmm23, dword ptr [rdx - 512]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x45,0xd7,0x52,0x72,0x80,0x7b]
          vminmaxps zmm22 {k7} {z}, zmm23, dword ptr [rdx - 512]{1to16}, 123

// CHECK: vminmaxsd xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x00,0x53,0xf0,0x7b]
          vminmaxsd xmm22, xmm23, xmm24, 123

// CHECK: vminmaxsd xmm22, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x10,0x53,0xf0,0x7b]
          vminmaxsd xmm22, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxsd xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x07,0x53,0xf0,0x7b]
          vminmaxsd xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxsd xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0xc5,0x97,0x53,0xf0,0x7b]
          vminmaxsd xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxsd xmm22, xmm23, qword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0xc5,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsd xmm22, xmm23, qword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxsd xmm22 {k7}, xmm23, qword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0xc5,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsd xmm22 {k7}, xmm23, qword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxsd xmm22, xmm23, qword ptr [rip], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxsd xmm22, xmm23, qword ptr [rip], 123

// CHECK: vminmaxsd xmm22, xmm23, qword ptr [2*rbp - 256], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x00,0x53,0x34,0x6d,0x00,0xff,0xff,0xff,0x7b]
          vminmaxsd xmm22, xmm23, qword ptr [2*rbp - 256], 123

// CHECK: vminmaxsd xmm22 {k7} {z}, xmm23, qword ptr [rcx + 1016], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxsd xmm22 {k7} {z}, xmm23, qword ptr [rcx + 1016], 123

// CHECK: vminmaxsd xmm22 {k7} {z}, xmm23, qword ptr [rdx - 1024], 123
// CHECK: encoding: [0x62,0xe3,0xc5,0x87,0x53,0x72,0x80,0x7b]
          vminmaxsd xmm22 {k7} {z}, xmm23, qword ptr [rdx - 1024], 123

// CHECK: vminmaxsh xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x00,0x53,0xf0,0x7b]
          vminmaxsh xmm22, xmm23, xmm24, 123

// CHECK: vminmaxsh xmm22, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x44,0x10,0x53,0xf0,0x7b]
          vminmaxsh xmm22, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxsh xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x44,0x07,0x53,0xf0,0x7b]
          vminmaxsh xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxsh xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x44,0x97,0x53,0xf0,0x7b]
          vminmaxsh xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxsh xmm22, xmm23, word ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x44,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsh xmm22, xmm23, word ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxsh xmm22 {k7}, xmm23, word ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x44,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsh xmm22 {k7}, xmm23, word ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxsh xmm22, xmm23, word ptr [rip], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxsh xmm22, xmm23, word ptr [rip], 123

// CHECK: vminmaxsh xmm22, xmm23, word ptr [2*rbp - 64], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x00,0x53,0x34,0x6d,0xc0,0xff,0xff,0xff,0x7b]
          vminmaxsh xmm22, xmm23, word ptr [2*rbp - 64], 123

// CHECK: vminmaxsh xmm22 {k7} {z}, xmm23, word ptr [rcx + 254], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxsh xmm22 {k7} {z}, xmm23, word ptr [rcx + 254], 123

// CHECK: vminmaxsh xmm22 {k7} {z}, xmm23, word ptr [rdx - 256], 123
// CHECK: encoding: [0x62,0xe3,0x44,0x87,0x53,0x72,0x80,0x7b]
          vminmaxsh xmm22 {k7} {z}, xmm23, word ptr [rdx - 256], 123

// CHECK: vminmaxss xmm22, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x00,0x53,0xf0,0x7b]
          vminmaxss xmm22, xmm23, xmm24, 123

// CHECK: vminmaxss xmm22, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x45,0x10,0x53,0xf0,0x7b]
          vminmaxss xmm22, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxss xmm22 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x83,0x45,0x07,0x53,0xf0,0x7b]
          vminmaxss xmm22 {k7}, xmm23, xmm24, 123

// CHECK: vminmaxss xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123
// CHECK: encoding: [0x62,0x83,0x45,0x97,0x53,0xf0,0x7b]
          vminmaxss xmm22 {k7} {z}, xmm23, xmm24, {sae}, 123

// CHECK: vminmaxss xmm22, xmm23, dword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x45,0x00,0x53,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vminmaxss xmm22, xmm23, dword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vminmaxss xmm22 {k7}, xmm23, dword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x45,0x07,0x53,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vminmaxss xmm22 {k7}, xmm23, dword ptr [r8 + 4*rax + 291], 123

// CHECK: vminmaxss xmm22, xmm23, dword ptr [rip], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x53,0x35,0x00,0x00,0x00,0x00,0x7b]
          vminmaxss xmm22, xmm23, dword ptr [rip], 123

// CHECK: vminmaxss xmm22, xmm23, dword ptr [2*rbp - 128], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x00,0x53,0x34,0x6d,0x80,0xff,0xff,0xff,0x7b]
          vminmaxss xmm22, xmm23, dword ptr [2*rbp - 128], 123

// CHECK: vminmaxss xmm22 {k7} {z}, xmm23, dword ptr [rcx + 508], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x53,0x71,0x7f,0x7b]
          vminmaxss xmm22 {k7} {z}, xmm23, dword ptr [rcx + 508], 123

// CHECK: vminmaxss xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512], 123
// CHECK: encoding: [0x62,0xe3,0x45,0x87,0x53,0x72,0x80,0x7b]
          vminmaxss xmm22 {k7} {z}, xmm23, dword ptr [rdx - 512], 123

