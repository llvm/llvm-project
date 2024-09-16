// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vaddnepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x58,0xf0]
          vaddnepbf16 ymm22, ymm23, ymm24

// CHECK: vaddnepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x58,0xf0]
          vaddnepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vaddnepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x58,0xf0]
          vaddnepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vaddnepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x58,0xf0]
          vaddnepbf16 zmm22, zmm23, zmm24

// CHECK: vaddnepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x58,0xf0]
          vaddnepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vaddnepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x58,0xf0]
          vaddnepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vaddnepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x58,0xf0]
          vaddnepbf16 xmm22, xmm23, xmm24

// CHECK: vaddnepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x58,0xf0]
          vaddnepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vaddnepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x58,0xf0]
          vaddnepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vaddnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vaddnepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vaddnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x58,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vaddnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vaddnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x58,0x71,0x7f]
          vaddnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vaddnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x58,0x72,0x80]
          vaddnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vaddnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vaddnepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vaddnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x58,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vaddnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vaddnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x58,0x71,0x7f]
          vaddnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vaddnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x58,0x72,0x80]
          vaddnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vaddnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vaddnepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddnepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vaddnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x58,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vaddnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vaddnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x58,0x71,0x7f]
          vaddnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vaddnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x58,0x72,0x80]
          vaddnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcmppbf16 k5, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x20,0xc2,0xe8,0x7b]
          vcmppbf16 k5, ymm23, ymm24, 123

// CHECK: vcmppbf16 k5 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x27,0xc2,0xe8,0x7b]
          vcmppbf16 k5 {k7}, ymm23, ymm24, 123

// CHECK: vcmppbf16 k5, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x00,0xc2,0xe8,0x7b]
          vcmppbf16 k5, xmm23, xmm24, 123

// CHECK: vcmppbf16 k5 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x07,0xc2,0xe8,0x7b]
          vcmppbf16 k5 {k7}, xmm23, xmm24, 123

// CHECK: vcmppbf16 k5, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x40,0xc2,0xe8,0x7b]
          vcmppbf16 k5, zmm23, zmm24, 123

// CHECK: vcmppbf16 k5 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x47,0xc2,0xe8,0x7b]
          vcmppbf16 k5 {k7}, zmm23, zmm24, 123

// CHECK: vcmppbf16 k5, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x40,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x47,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmppbf16 k5, zmm23, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x50,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16 k5, zmm23, word ptr [rip]{1to32}, 123

// CHECK: vcmppbf16 k5, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x40,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmppbf16 k5, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vcmppbf16 k5 {k7}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x47,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vcmppbf16 k5 {k7}, zmm23, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x57,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, zmm23, word ptr [rdx - 256]{1to32}, 123

// CHECK: vcmppbf16 k5, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x00,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x07,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmppbf16 k5, xmm23, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x10,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16 k5, xmm23, word ptr [rip]{1to8}, 123

// CHECK: vcmppbf16 k5, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x00,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmppbf16 k5, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vcmppbf16 k5 {k7}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x07,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vcmppbf16 k5 {k7}, xmm23, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x17,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, xmm23, word ptr [rdx - 256]{1to8}, 123

// CHECK: vcmppbf16 k5, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x20,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmppbf16 k5, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmppbf16 k5 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x27,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmppbf16 k5 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmppbf16 k5, ymm23, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x30,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmppbf16 k5, ymm23, word ptr [rip]{1to16}, 123

// CHECK: vcmppbf16 k5, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x20,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmppbf16 k5, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vcmppbf16 k5 {k7}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x27,0xc2,0x69,0x7f,0x7b]
          vcmppbf16 k5 {k7}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vcmppbf16 k5 {k7}, ymm23, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x37,0xc2,0x6a,0x80,0x7b]
          vcmppbf16 k5 {k7}, ymm23, word ptr [rdx - 256]{1to16}, 123

// CHECK: vcomsbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xf7]
          vcomsbf16 xmm22, xmm23

// CHECK: vcomsbf16 xmm22, word ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcomsbf16 xmm22, word ptr [rbp + 8*r14 + 268435456]

// CHECK: vcomsbf16 xmm22, word ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x08,0x2f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcomsbf16 xmm22, word ptr [r8 + 4*rax + 291]

// CHECK: vcomsbf16 xmm22, word ptr [rip]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x35,0x00,0x00,0x00,0x00]
          vcomsbf16 xmm22, word ptr [rip]

// CHECK: vcomsbf16 xmm22, word ptr [2*rbp - 64]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x34,0x6d,0xc0,0xff,0xff,0xff]
          vcomsbf16 xmm22, word ptr [2*rbp - 64]

// CHECK: vcomsbf16 xmm22, word ptr [rcx + 254]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x71,0x7f]
          vcomsbf16 xmm22, word ptr [rcx + 254]

// CHECK: vcomsbf16 xmm22, word ptr [rdx - 256]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x72,0x80]
          vcomsbf16 xmm22, word ptr [rdx - 256]

// CHECK: vdivnepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5e,0xf0]
          vdivnepbf16 ymm22, ymm23, ymm24

// CHECK: vdivnepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5e,0xf0]
          vdivnepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vdivnepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5e,0xf0]
          vdivnepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vdivnepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5e,0xf0]
          vdivnepbf16 zmm22, zmm23, zmm24

// CHECK: vdivnepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5e,0xf0]
          vdivnepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vdivnepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5e,0xf0]
          vdivnepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vdivnepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5e,0xf0]
          vdivnepbf16 xmm22, xmm23, xmm24

// CHECK: vdivnepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5e,0xf0]
          vdivnepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vdivnepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5e,0xf0]
          vdivnepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vdivnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vdivnepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vdivnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vdivnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vdivnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5e,0x71,0x7f]
          vdivnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vdivnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5e,0x72,0x80]
          vdivnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vdivnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vdivnepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vdivnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vdivnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vdivnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5e,0x71,0x7f]
          vdivnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vdivnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5e,0x72,0x80]
          vdivnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vdivnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vdivnepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivnepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vdivnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vdivnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vdivnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5e,0x71,0x7f]
          vdivnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vdivnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5e,0x72,0x80]
          vdivnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd132nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x98,0xf0]
          vfmadd132nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmadd132nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x98,0xf0]
          vfmadd132nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x98,0xf0]
          vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd132nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x98,0xf0]
          vfmadd132nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmadd132nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x98,0xf0]
          vfmadd132nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x98,0xf0]
          vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd132nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x98,0xf0]
          vfmadd132nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmadd132nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x98,0xf0]
          vfmadd132nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x98,0xf0]
          vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x98,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x98,0x71,0x7f]
          vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x98,0x72,0x80]
          vfmadd132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x98,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x98,0x71,0x7f]
          vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x98,0x72,0x80]
          vfmadd132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x98,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x98,0x71,0x7f]
          vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x98,0x72,0x80]
          vfmadd132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd213nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xa8,0xf0]
          vfmadd213nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmadd213nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xa8,0xf0]
          vfmadd213nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xa8,0xf0]
          vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd213nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xa8,0xf0]
          vfmadd213nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmadd213nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xa8,0xf0]
          vfmadd213nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xa8,0xf0]
          vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd213nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xa8,0xf0]
          vfmadd213nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmadd213nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xa8,0xf0]
          vfmadd213nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xa8,0xf0]
          vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xa8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xa8,0x71,0x7f]
          vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xa8,0x72,0x80]
          vfmadd213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xa8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xa8,0x71,0x7f]
          vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xa8,0x72,0x80]
          vfmadd213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xa8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xa8,0x71,0x7f]
          vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xa8,0x72,0x80]
          vfmadd213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd231nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xb8,0xf0]
          vfmadd231nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmadd231nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xb8,0xf0]
          vfmadd231nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xb8,0xf0]
          vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd231nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xb8,0xf0]
          vfmadd231nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmadd231nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xb8,0xf0]
          vfmadd231nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xb8,0xf0]
          vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd231nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xb8,0xf0]
          vfmadd231nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmadd231nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xb8,0xf0]
          vfmadd231nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xb8,0xf0]
          vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xb8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xb8,0x71,0x7f]
          vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xb8,0x72,0x80]
          vfmadd231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xb8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xb8,0x71,0x7f]
          vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xb8,0x72,0x80]
          vfmadd231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xb8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xb8,0x71,0x7f]
          vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xb8,0x72,0x80]
          vfmadd231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub132nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9a,0xf0]
          vfmsub132nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmsub132nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9a,0xf0]
          vfmsub132nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9a,0xf0]
          vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub132nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9a,0xf0]
          vfmsub132nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmsub132nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9a,0xf0]
          vfmsub132nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9a,0xf0]
          vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub132nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9a,0xf0]
          vfmsub132nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmsub132nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9a,0xf0]
          vfmsub132nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9a,0xf0]
          vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9a,0x71,0x7f]
          vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9a,0x72,0x80]
          vfmsub132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9a,0x71,0x7f]
          vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9a,0x72,0x80]
          vfmsub132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9a,0x71,0x7f]
          vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9a,0x72,0x80]
          vfmsub132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub213nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xaa,0xf0]
          vfmsub213nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmsub213nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xaa,0xf0]
          vfmsub213nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xaa,0xf0]
          vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub213nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xaa,0xf0]
          vfmsub213nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmsub213nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xaa,0xf0]
          vfmsub213nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xaa,0xf0]
          vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub213nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xaa,0xf0]
          vfmsub213nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmsub213nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xaa,0xf0]
          vfmsub213nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xaa,0xf0]
          vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xaa,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xaa,0x71,0x7f]
          vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xaa,0x72,0x80]
          vfmsub213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xaa,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xaa,0x71,0x7f]
          vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xaa,0x72,0x80]
          vfmsub213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xaa,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xaa,0x71,0x7f]
          vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xaa,0x72,0x80]
          vfmsub213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub231nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xba,0xf0]
          vfmsub231nepbf16 ymm22, ymm23, ymm24

// CHECK: vfmsub231nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xba,0xf0]
          vfmsub231nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xba,0xf0]
          vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub231nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xba,0xf0]
          vfmsub231nepbf16 zmm22, zmm23, zmm24

// CHECK: vfmsub231nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xba,0xf0]
          vfmsub231nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xba,0xf0]
          vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub231nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xba,0xf0]
          vfmsub231nepbf16 xmm22, xmm23, xmm24

// CHECK: vfmsub231nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xba,0xf0]
          vfmsub231nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xba,0xf0]
          vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xba,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xba,0x71,0x7f]
          vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xba,0x72,0x80]
          vfmsub231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xba,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xba,0x71,0x7f]
          vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xba,0x72,0x80]
          vfmsub231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xba,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xba,0x71,0x7f]
          vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xba,0x72,0x80]
          vfmsub231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd132nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9c,0xf0]
          vfnmadd132nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd132nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9c,0xf0]
          vfnmadd132nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9c,0xf0]
          vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd132nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9c,0xf0]
          vfnmadd132nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd132nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9c,0xf0]
          vfnmadd132nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9c,0xf0]
          vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd132nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9c,0xf0]
          vfnmadd132nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd132nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9c,0xf0]
          vfnmadd132nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9c,0xf0]
          vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9c,0x71,0x7f]
          vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9c,0x72,0x80]
          vfnmadd132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9c,0x71,0x7f]
          vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9c,0x72,0x80]
          vfnmadd132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9c,0x71,0x7f]
          vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9c,0x72,0x80]
          vfnmadd132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd213nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xac,0xf0]
          vfnmadd213nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd213nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xac,0xf0]
          vfnmadd213nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xac,0xf0]
          vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd213nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xac,0xf0]
          vfnmadd213nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd213nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xac,0xf0]
          vfnmadd213nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xac,0xf0]
          vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd213nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xac,0xf0]
          vfnmadd213nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd213nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xac,0xf0]
          vfnmadd213nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xac,0xf0]
          vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xac,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xac,0x71,0x7f]
          vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xac,0x72,0x80]
          vfnmadd213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xac,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xac,0x71,0x7f]
          vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xac,0x72,0x80]
          vfnmadd213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xac,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xac,0x71,0x7f]
          vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xac,0x72,0x80]
          vfnmadd213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd231nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbc,0xf0]
          vfnmadd231nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd231nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbc,0xf0]
          vfnmadd231nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbc,0xf0]
          vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd231nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbc,0xf0]
          vfnmadd231nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd231nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbc,0xf0]
          vfnmadd231nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbc,0xf0]
          vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd231nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbc,0xf0]
          vfnmadd231nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd231nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbc,0xf0]
          vfnmadd231nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbc,0xf0]
          vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbc,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbc,0x71,0x7f]
          vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbc,0x72,0x80]
          vfnmadd231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbc,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbc,0x71,0x7f]
          vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbc,0x72,0x80]
          vfnmadd231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbc,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbc,0x71,0x7f]
          vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbc,0x72,0x80]
          vfnmadd231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub132nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9e,0xf0]
          vfnmsub132nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub132nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9e,0xf0]
          vfnmsub132nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9e,0xf0]
          vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub132nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9e,0xf0]
          vfnmsub132nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub132nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9e,0xf0]
          vfnmsub132nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9e,0xf0]
          vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub132nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9e,0xf0]
          vfnmsub132nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub132nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9e,0xf0]
          vfnmsub132nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9e,0xf0]
          vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9e,0x71,0x7f]
          vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9e,0x72,0x80]
          vfnmsub132nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9e,0x71,0x7f]
          vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9e,0x72,0x80]
          vfnmsub132nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9e,0x71,0x7f]
          vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9e,0x72,0x80]
          vfnmsub132nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub213nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xae,0xf0]
          vfnmsub213nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub213nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xae,0xf0]
          vfnmsub213nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xae,0xf0]
          vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub213nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xae,0xf0]
          vfnmsub213nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub213nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xae,0xf0]
          vfnmsub213nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xae,0xf0]
          vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub213nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xae,0xf0]
          vfnmsub213nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub213nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xae,0xf0]
          vfnmsub213nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xae,0xf0]
          vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xae,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xae,0x71,0x7f]
          vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xae,0x72,0x80]
          vfnmsub213nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xae,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xae,0x71,0x7f]
          vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xae,0x72,0x80]
          vfnmsub213nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xae,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xae,0x71,0x7f]
          vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xae,0x72,0x80]
          vfnmsub213nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub231nepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbe,0xf0]
          vfnmsub231nepbf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub231nepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbe,0xf0]
          vfnmsub231nepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbe,0xf0]
          vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub231nepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbe,0xf0]
          vfnmsub231nepbf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub231nepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbe,0xf0]
          vfnmsub231nepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbe,0xf0]
          vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub231nepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbe,0xf0]
          vfnmsub231nepbf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub231nepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbe,0xf0]
          vfnmsub231nepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbe,0xf0]
          vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbe,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231nepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbe,0x71,0x7f]
          vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbe,0x72,0x80]
          vfnmsub231nepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbe,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231nepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbe,0x71,0x7f]
          vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbe,0x72,0x80]
          vfnmsub231nepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231nepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231nepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231nepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbe,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231nepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbe,0x71,0x7f]
          vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbe,0x72,0x80]
          vfnmsub231nepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfpclasspbf16 k5, zmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x48,0x66,0xef,0x7b]
          vfpclasspbf16 k5, zmm23, 123

// CHECK: vfpclasspbf16 k5 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x4f,0x66,0xef,0x7b]
          vfpclasspbf16 k5 {k7}, zmm23, 123

// CHECK: vfpclasspbf16 k5, ymm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x28,0x66,0xef,0x7b]
          vfpclasspbf16 k5, ymm23, 123

// CHECK: vfpclasspbf16 k5 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x2f,0x66,0xef,0x7b]
          vfpclasspbf16 k5 {k7}, ymm23, 123

// CHECK: vfpclasspbf16 k5, xmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xef,0x7b]
          vfpclasspbf16 k5, xmm23, 123

// CHECK: vfpclasspbf16 k5 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x0f,0x66,0xef,0x7b]
          vfpclasspbf16 k5 {k7}, xmm23, 123

// CHECK: vfpclasspbf16 k5, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclasspbf16 k5, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vfpclasspbf16 k5 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x7f,0x0f,0x66,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vfpclasspbf16 k5 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vfpclasspbf16 k5, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16 k5, word ptr [rip]{1to8}, 123

// CHECK: vfpclasspbf16 k5, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclasspbf16 k5, xmmword ptr [2*rbp - 512], 123

// CHECK: vfpclasspbf16 k5 {k7}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, xmmword ptr [rcx + 2032], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vfpclasspbf16 k5, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16 k5, word ptr [rip]{1to16}, 123

// CHECK: vfpclasspbf16 k5, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclasspbf16 k5, ymmword ptr [2*rbp - 1024], 123

// CHECK: vfpclasspbf16 k5 {k7}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, ymmword ptr [rcx + 4064], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vfpclasspbf16 k5, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclasspbf16 k5, word ptr [rip]{1to32}, 123

// CHECK: vfpclasspbf16 k5, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclasspbf16 k5, zmmword ptr [2*rbp - 2048], 123

// CHECK: vfpclasspbf16 k5 {k7}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclasspbf16 k5 {k7}, zmmword ptr [rcx + 8128], 123

// CHECK: vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclasspbf16 k5 {k7}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vgetexppbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xf7]
          vgetexppbf16 xmm22, xmm23

// CHECK: vgetexppbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x42,0xf7]
          vgetexppbf16 xmm22 {k7}, xmm23

// CHECK: vgetexppbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x42,0xf7]
          vgetexppbf16 xmm22 {k7} {z}, xmm23

// CHECK: vgetexppbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xf7]
          vgetexppbf16 zmm22, zmm23

// CHECK: vgetexppbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x42,0xf7]
          vgetexppbf16 zmm22 {k7}, zmm23

// CHECK: vgetexppbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x42,0xf7]
          vgetexppbf16 zmm22 {k7} {z}, zmm23

// CHECK: vgetexppbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xf7]
          vgetexppbf16 ymm22, ymm23

// CHECK: vgetexppbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x42,0xf7]
          vgetexppbf16 ymm22 {k7}, ymm23

// CHECK: vgetexppbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x42,0xf7]
          vgetexppbf16 ymm22 {k7} {z}, ymm23

// CHECK: vgetexppbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexppbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexppbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vgetexppbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x42,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vgetexppbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vgetexppbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x42,0x71,0x7f]
          vgetexppbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vgetexppbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x42,0x72,0x80]
          vgetexppbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vgetexppbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexppbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexppbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vgetexppbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x42,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vgetexppbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vgetexppbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x42,0x71,0x7f]
          vgetexppbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vgetexppbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x42,0x72,0x80]
          vgetexppbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vgetexppbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexppbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexppbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexppbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexppbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexppbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vgetexppbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x42,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vgetexppbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vgetexppbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x42,0x71,0x7f]
          vgetexppbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vgetexppbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x42,0x72,0x80]
          vgetexppbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vgetmantpbf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xf7,0x7b]
          vgetmantpbf16 zmm22, zmm23, 123

// CHECK: vgetmantpbf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x26,0xf7,0x7b]
          vgetmantpbf16 zmm22 {k7}, zmm23, 123

// CHECK: vgetmantpbf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x26,0xf7,0x7b]
          vgetmantpbf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vgetmantpbf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xf7,0x7b]
          vgetmantpbf16 ymm22, ymm23, 123

// CHECK: vgetmantpbf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x26,0xf7,0x7b]
          vgetmantpbf16 ymm22 {k7}, ymm23, 123

// CHECK: vgetmantpbf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x26,0xf7,0x7b]
          vgetmantpbf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vgetmantpbf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xf7,0x7b]
          vgetmantpbf16 xmm22, xmm23, 123

// CHECK: vgetmantpbf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x26,0xf7,0x7b]
          vgetmantpbf16 xmm22 {k7}, xmm23, 123

// CHECK: vgetmantpbf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x26,0xf7,0x7b]
          vgetmantpbf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vgetmantpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantpbf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vgetmantpbf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x26,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantpbf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vgetmantpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vgetmantpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vgetmantpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantpbf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vgetmantpbf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x26,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantpbf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vgetmantpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vgetmantpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vgetmantpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantpbf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantpbf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vgetmantpbf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x26,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantpbf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vgetmantpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x26,0x71,0x7f,0x7b]
          vgetmantpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vgetmantpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vmaxpbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5f,0xf0]
          vmaxpbf16 ymm22, ymm23, ymm24

// CHECK: vmaxpbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5f,0xf0]
          vmaxpbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vmaxpbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5f,0xf0]
          vmaxpbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vmaxpbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5f,0xf0]
          vmaxpbf16 zmm22, zmm23, zmm24

// CHECK: vmaxpbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5f,0xf0]
          vmaxpbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vmaxpbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5f,0xf0]
          vmaxpbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vmaxpbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5f,0xf0]
          vmaxpbf16 xmm22, xmm23, xmm24

// CHECK: vmaxpbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5f,0xf0]
          vmaxpbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vmaxpbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5f,0xf0]
          vmaxpbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vmaxpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxpbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vmaxpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5f,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmaxpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vmaxpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5f,0x71,0x7f]
          vmaxpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vmaxpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5f,0x72,0x80]
          vmaxpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vmaxpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxpbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vmaxpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5f,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmaxpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vmaxpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5f,0x71,0x7f]
          vmaxpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vmaxpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5f,0x72,0x80]
          vmaxpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vmaxpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxpbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxpbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vmaxpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5f,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmaxpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vmaxpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5f,0x71,0x7f]
          vmaxpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vmaxpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5f,0x72,0x80]
          vmaxpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vminpbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5d,0xf0]
          vminpbf16 ymm22, ymm23, ymm24

// CHECK: vminpbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5d,0xf0]
          vminpbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vminpbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5d,0xf0]
          vminpbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vminpbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5d,0xf0]
          vminpbf16 zmm22, zmm23, zmm24

// CHECK: vminpbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5d,0xf0]
          vminpbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vminpbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5d,0xf0]
          vminpbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vminpbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5d,0xf0]
          vminpbf16 xmm22, xmm23, xmm24

// CHECK: vminpbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5d,0xf0]
          vminpbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vminpbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5d,0xf0]
          vminpbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vminpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vminpbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vminpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vminpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vminpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5d,0x71,0x7f]
          vminpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vminpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5d,0x72,0x80]
          vminpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vminpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vminpbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vminpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vminpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vminpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5d,0x71,0x7f]
          vminpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vminpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5d,0x72,0x80]
          vminpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vminpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vminpbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminpbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vminpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vminpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vminpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5d,0x71,0x7f]
          vminpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vminpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5d,0x72,0x80]
          vminpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vmulnepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x59,0xf0]
          vmulnepbf16 ymm22, ymm23, ymm24

// CHECK: vmulnepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x59,0xf0]
          vmulnepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vmulnepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x59,0xf0]
          vmulnepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vmulnepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x59,0xf0]
          vmulnepbf16 zmm22, zmm23, zmm24

// CHECK: vmulnepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x59,0xf0]
          vmulnepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vmulnepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x59,0xf0]
          vmulnepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vmulnepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x59,0xf0]
          vmulnepbf16 xmm22, xmm23, xmm24

// CHECK: vmulnepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x59,0xf0]
          vmulnepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vmulnepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x59,0xf0]
          vmulnepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vmulnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vmulnepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vmulnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x59,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmulnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vmulnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x59,0x71,0x7f]
          vmulnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vmulnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x59,0x72,0x80]
          vmulnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vmulnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vmulnepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vmulnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x59,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmulnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vmulnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x59,0x71,0x7f]
          vmulnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vmulnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x59,0x72,0x80]
          vmulnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vmulnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vmulnepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulnepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vmulnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x59,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmulnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vmulnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x59,0x71,0x7f]
          vmulnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vmulnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x59,0x72,0x80]
          vmulnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vrcppbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xf7]
          vrcppbf16 xmm22, xmm23

// CHECK: vrcppbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4c,0xf7]
          vrcppbf16 xmm22 {k7}, xmm23

// CHECK: vrcppbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4c,0xf7]
          vrcppbf16 xmm22 {k7} {z}, xmm23

// CHECK: vrcppbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xf7]
          vrcppbf16 zmm22, zmm23

// CHECK: vrcppbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4c,0xf7]
          vrcppbf16 zmm22 {k7}, zmm23

// CHECK: vrcppbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4c,0xf7]
          vrcppbf16 zmm22 {k7} {z}, zmm23

// CHECK: vrcppbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xf7]
          vrcppbf16 ymm22, ymm23

// CHECK: vrcppbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4c,0xf7]
          vrcppbf16 ymm22 {k7}, ymm23

// CHECK: vrcppbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4c,0xf7]
          vrcppbf16 ymm22 {k7} {z}, ymm23

// CHECK: vrcppbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcppbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vrcppbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vrcppbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrcppbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vrcppbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4c,0x71,0x7f]
          vrcppbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vrcppbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4c,0x72,0x80]
          vrcppbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrcppbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcppbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vrcppbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vrcppbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrcppbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vrcppbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4c,0x71,0x7f]
          vrcppbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vrcppbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4c,0x72,0x80]
          vrcppbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vrcppbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcppbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcppbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcppbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vrcppbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcppbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vrcppbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrcppbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vrcppbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4c,0x71,0x7f]
          vrcppbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vrcppbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4c,0x72,0x80]
          vrcppbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vreducenepbf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xf7,0x7b]
          vreducenepbf16 zmm22, zmm23, 123

// CHECK: vreducenepbf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x56,0xf7,0x7b]
          vreducenepbf16 zmm22 {k7}, zmm23, 123

// CHECK: vreducenepbf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x56,0xf7,0x7b]
          vreducenepbf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vreducenepbf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xf7,0x7b]
          vreducenepbf16 ymm22, ymm23, 123

// CHECK: vreducenepbf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x56,0xf7,0x7b]
          vreducenepbf16 ymm22 {k7}, ymm23, 123

// CHECK: vreducenepbf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x56,0xf7,0x7b]
          vreducenepbf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vreducenepbf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xf7,0x7b]
          vreducenepbf16 xmm22, xmm23, 123

// CHECK: vreducenepbf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x56,0xf7,0x7b]
          vreducenepbf16 xmm22 {k7}, xmm23, 123

// CHECK: vreducenepbf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x56,0xf7,0x7b]
          vreducenepbf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vreducenepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducenepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducenepbf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vreducenepbf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x56,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducenepbf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vreducenepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x56,0x71,0x7f,0x7b]
          vreducenepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vreducenepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x56,0x72,0x80,0x7b]
          vreducenepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vreducenepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducenepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducenepbf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vreducenepbf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x56,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducenepbf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vreducenepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x56,0x71,0x7f,0x7b]
          vreducenepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vreducenepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x56,0x72,0x80,0x7b]
          vreducenepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vreducenepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducenepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducenepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducenepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducenepbf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducenepbf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vreducenepbf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x56,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducenepbf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vreducenepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x56,0x71,0x7f,0x7b]
          vreducenepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vreducenepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x56,0x72,0x80,0x7b]
          vreducenepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vrndscalenepbf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xf7,0x7b]
          vrndscalenepbf16 zmm22, zmm23, 123

// CHECK: vrndscalenepbf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x08,0xf7,0x7b]
          vrndscalenepbf16 zmm22 {k7}, zmm23, 123

// CHECK: vrndscalenepbf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x08,0xf7,0x7b]
          vrndscalenepbf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vrndscalenepbf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xf7,0x7b]
          vrndscalenepbf16 ymm22, ymm23, 123

// CHECK: vrndscalenepbf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x08,0xf7,0x7b]
          vrndscalenepbf16 ymm22 {k7}, ymm23, 123

// CHECK: vrndscalenepbf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x08,0xf7,0x7b]
          vrndscalenepbf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vrndscalenepbf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xf7,0x7b]
          vrndscalenepbf16 xmm22, xmm23, 123

// CHECK: vrndscalenepbf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x08,0xf7,0x7b]
          vrndscalenepbf16 xmm22 {k7}, xmm23, 123

// CHECK: vrndscalenepbf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x08,0xf7,0x7b]
          vrndscalenepbf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vrndscalenepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalenepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalenepbf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vrndscalenepbf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x08,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalenepbf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vrndscalenepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vrndscalenepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vrndscalenepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalenepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalenepbf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vrndscalenepbf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x08,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalenepbf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vrndscalenepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vrndscalenepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vrndscalenepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalenepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalenepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalenepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalenepbf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalenepbf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vrndscalenepbf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x08,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalenepbf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vrndscalenepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x08,0x71,0x7f,0x7b]
          vrndscalenepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vrndscalenepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x08,0x72,0x80,0x7b]
          vrndscalenepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vrsqrtpbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xf7]
          vrsqrtpbf16 xmm22, xmm23

// CHECK: vrsqrtpbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4e,0xf7]
          vrsqrtpbf16 xmm22 {k7}, xmm23

// CHECK: vrsqrtpbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4e,0xf7]
          vrsqrtpbf16 xmm22 {k7} {z}, xmm23

// CHECK: vrsqrtpbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xf7]
          vrsqrtpbf16 zmm22, zmm23

// CHECK: vrsqrtpbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4e,0xf7]
          vrsqrtpbf16 zmm22 {k7}, zmm23

// CHECK: vrsqrtpbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4e,0xf7]
          vrsqrtpbf16 zmm22 {k7} {z}, zmm23

// CHECK: vrsqrtpbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xf7]
          vrsqrtpbf16 ymm22, ymm23

// CHECK: vrsqrtpbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4e,0xf7]
          vrsqrtpbf16 ymm22 {k7}, ymm23

// CHECK: vrsqrtpbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4e,0xf7]
          vrsqrtpbf16 ymm22 {k7} {z}, ymm23

// CHECK: vrsqrtpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtpbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vrsqrtpbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtpbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vrsqrtpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4e,0x71,0x7f]
          vrsqrtpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vrsqrtpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4e,0x72,0x80]
          vrsqrtpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrsqrtpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtpbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vrsqrtpbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtpbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vrsqrtpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4e,0x71,0x7f]
          vrsqrtpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vrsqrtpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4e,0x72,0x80]
          vrsqrtpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vrsqrtpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtpbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtpbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vrsqrtpbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtpbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vrsqrtpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4e,0x71,0x7f]
          vrsqrtpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vrsqrtpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4e,0x72,0x80]
          vrsqrtpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vscalefpbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x2c,0xf0]
          vscalefpbf16 ymm22, ymm23, ymm24

// CHECK: vscalefpbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x2c,0xf0]
          vscalefpbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vscalefpbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x2c,0xf0]
          vscalefpbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vscalefpbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x2c,0xf0]
          vscalefpbf16 zmm22, zmm23, zmm24

// CHECK: vscalefpbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x2c,0xf0]
          vscalefpbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vscalefpbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x2c,0xf0]
          vscalefpbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vscalefpbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x2c,0xf0]
          vscalefpbf16 xmm22, xmm23, xmm24

// CHECK: vscalefpbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x2c,0xf0]
          vscalefpbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vscalefpbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x2c,0xf0]
          vscalefpbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vscalefpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefpbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vscalefpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x2c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vscalefpbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vscalefpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x2c,0x71,0x7f]
          vscalefpbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vscalefpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x2c,0x72,0x80]
          vscalefpbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vscalefpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefpbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vscalefpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x2c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vscalefpbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vscalefpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x2c,0x71,0x7f]
          vscalefpbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vscalefpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x2c,0x72,0x80]
          vscalefpbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vscalefpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefpbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefpbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefpbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefpbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vscalefpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x2c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vscalefpbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vscalefpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x2c,0x71,0x7f]
          vscalefpbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vscalefpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x2c,0x72,0x80]
          vscalefpbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtnepbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xf7]
          vsqrtnepbf16 xmm22, xmm23

// CHECK: vsqrtnepbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x51,0xf7]
          vsqrtnepbf16 xmm22 {k7}, xmm23

// CHECK: vsqrtnepbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x51,0xf7]
          vsqrtnepbf16 xmm22 {k7} {z}, xmm23

// CHECK: vsqrtnepbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xf7]
          vsqrtnepbf16 zmm22, zmm23

// CHECK: vsqrtnepbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x51,0xf7]
          vsqrtnepbf16 zmm22 {k7}, zmm23

// CHECK: vsqrtnepbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x51,0xf7]
          vsqrtnepbf16 zmm22 {k7} {z}, zmm23

// CHECK: vsqrtnepbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xf7]
          vsqrtnepbf16 ymm22, ymm23

// CHECK: vsqrtnepbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x51,0xf7]
          vsqrtnepbf16 ymm22 {k7}, ymm23

// CHECK: vsqrtnepbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x51,0xf7]
          vsqrtnepbf16 ymm22 {k7} {z}, ymm23

// CHECK: vsqrtnepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtnepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtnepbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vsqrtnepbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtnepbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vsqrtnepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x51,0x71,0x7f]
          vsqrtnepbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vsqrtnepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x51,0x72,0x80]
          vsqrtnepbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtnepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtnepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtnepbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vsqrtnepbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtnepbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vsqrtnepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x51,0x71,0x7f]
          vsqrtnepbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vsqrtnepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x51,0x72,0x80]
          vsqrtnepbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vsqrtnepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtnepbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtnepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtnepbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtnepbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtnepbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vsqrtnepbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtnepbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vsqrtnepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x51,0x71,0x7f]
          vsqrtnepbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vsqrtnepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x51,0x72,0x80]
          vsqrtnepbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vsubnepbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5c,0xf0]
          vsubnepbf16 ymm22, ymm23, ymm24

// CHECK: vsubnepbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5c,0xf0]
          vsubnepbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vsubnepbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5c,0xf0]
          vsubnepbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vsubnepbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5c,0xf0]
          vsubnepbf16 zmm22, zmm23, zmm24

// CHECK: vsubnepbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5c,0xf0]
          vsubnepbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vsubnepbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5c,0xf0]
          vsubnepbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vsubnepbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5c,0xf0]
          vsubnepbf16 xmm22, xmm23, xmm24

// CHECK: vsubnepbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5c,0xf0]
          vsubnepbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vsubnepbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5c,0xf0]
          vsubnepbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vsubnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vsubnepbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vsubnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsubnepbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vsubnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5c,0x71,0x7f]
          vsubnepbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vsubnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5c,0x72,0x80]
          vsubnepbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vsubnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsubnepbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vsubnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsubnepbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vsubnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5c,0x71,0x7f]
          vsubnepbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vsubnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5c,0x72,0x80]
          vsubnepbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vsubnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubnepbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubnepbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsubnepbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubnepbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vsubnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsubnepbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vsubnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5c,0x71,0x7f]
          vsubnepbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vsubnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5c,0x72,0x80]
          vsubnepbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

