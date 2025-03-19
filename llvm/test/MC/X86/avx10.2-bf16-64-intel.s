// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vaddbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x58,0xf0]
          vaddbf16 ymm22, ymm23, ymm24

// CHECK: vaddbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x58,0xf0]
          vaddbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vaddbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x58,0xf0]
          vaddbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vaddbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x58,0xf0]
          vaddbf16 zmm22, zmm23, zmm24

// CHECK: vaddbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x58,0xf0]
          vaddbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vaddbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x58,0xf0]
          vaddbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vaddbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x58,0xf0]
          vaddbf16 xmm22, xmm23, xmm24

// CHECK: vaddbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x58,0xf0]
          vaddbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vaddbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x58,0xf0]
          vaddbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vaddbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vaddbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vaddbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x58,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vaddbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vaddbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x58,0x71,0x7f]
          vaddbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vaddbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x58,0x72,0x80]
          vaddbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vaddbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vaddbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vaddbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x58,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vaddbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vaddbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x58,0x71,0x7f]
          vaddbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vaddbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x58,0x72,0x80]
          vaddbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vaddbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x58,0xb4,0x80,0x23,0x01,0x00,0x00]
          vaddbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vaddbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x58,0x35,0x00,0x00,0x00,0x00]
          vaddbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vaddbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x58,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vaddbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vaddbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x58,0x71,0x7f]
          vaddbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vaddbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x58,0x72,0x80]
          vaddbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vcmpbf16 k5, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x20,0xc2,0xe8,0x7b]
          vcmpbf16 k5, ymm23, ymm24, 123

// CHECK: vcmpbf16 k5 {k7}, ymm23, ymm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x27,0xc2,0xe8,0x7b]
          vcmpbf16 k5 {k7}, ymm23, ymm24, 123

// CHECK: vcmpbf16 k5, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x00,0xc2,0xe8,0x7b]
          vcmpbf16 k5, xmm23, xmm24, 123

// CHECK: vcmpbf16 k5 {k7}, xmm23, xmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x07,0xc2,0xe8,0x7b]
          vcmpbf16 k5 {k7}, xmm23, xmm24, 123

// CHECK: vcmpbf16 k5, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x40,0xc2,0xe8,0x7b]
          vcmpbf16 k5, zmm23, zmm24, 123

// CHECK: vcmpbf16 k5 {k7}, zmm23, zmm24, 123
// CHECK: encoding: [0x62,0x93,0x47,0x47,0xc2,0xe8,0x7b]
          vcmpbf16 k5 {k7}, zmm23, zmm24, 123

// CHECK: vcmpbf16 k5, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x40,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, zmm23, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x47,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmpbf16 k5, zmm23, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x50,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16 k5, zmm23, word ptr [rip]{1to32}, 123

// CHECK: vcmpbf16 k5, zmm23, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x40,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmpbf16 k5, zmm23, zmmword ptr [2*rbp - 2048], 123

// CHECK: vcmpbf16 k5 {k7}, zmm23, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x47,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, zmm23, zmmword ptr [rcx + 8128], 123

// CHECK: vcmpbf16 k5 {k7}, zmm23, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x57,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, zmm23, word ptr [rdx - 256]{1to32}, 123

// CHECK: vcmpbf16 k5, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x00,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, xmm23, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x07,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmpbf16 k5, xmm23, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x10,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16 k5, xmm23, word ptr [rip]{1to8}, 123

// CHECK: vcmpbf16 k5, xmm23, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x00,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmpbf16 k5, xmm23, xmmword ptr [2*rbp - 512], 123

// CHECK: vcmpbf16 k5 {k7}, xmm23, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x07,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, xmm23, xmmword ptr [rcx + 2032], 123

// CHECK: vcmpbf16 k5 {k7}, xmm23, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x17,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, xmm23, word ptr [rdx - 256]{1to8}, 123

// CHECK: vcmpbf16 k5, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x47,0x20,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16 k5, ymm23, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vcmpbf16 k5 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x47,0x27,0xc2,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16 k5 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vcmpbf16 k5, ymm23, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x30,0xc2,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vcmpbf16 k5, ymm23, word ptr [rip]{1to16}, 123

// CHECK: vcmpbf16 k5, ymm23, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x20,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmpbf16 k5, ymm23, ymmword ptr [2*rbp - 1024], 123

// CHECK: vcmpbf16 k5 {k7}, ymm23, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x47,0x27,0xc2,0x69,0x7f,0x7b]
          vcmpbf16 k5 {k7}, ymm23, ymmword ptr [rcx + 4064], 123

// CHECK: vcmpbf16 k5 {k7}, ymm23, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x47,0x37,0xc2,0x6a,0x80,0x7b]
          vcmpbf16 k5 {k7}, ymm23, word ptr [rdx - 256]{1to16}, 123

// CHECK: vcomisbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xf7]
          vcomisbf16 xmm22, xmm23

// CHECK: vcomisbf16 xmm22, word ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x2f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcomisbf16 xmm22, word ptr [rbp + 8*r14 + 268435456]

// CHECK: vcomisbf16 xmm22, word ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x08,0x2f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcomisbf16 xmm22, word ptr [r8 + 4*rax + 291]

// CHECK: vcomisbf16 xmm22, word ptr [rip]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x35,0x00,0x00,0x00,0x00]
          vcomisbf16 xmm22, word ptr [rip]

// CHECK: vcomisbf16 xmm22, word ptr [2*rbp - 64]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x34,0x6d,0xc0,0xff,0xff,0xff]
          vcomisbf16 xmm22, word ptr [2*rbp - 64]

// CHECK: vcomisbf16 xmm22, word ptr [rcx + 254]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x71,0x7f]
          vcomisbf16 xmm22, word ptr [rcx + 254]

// CHECK: vcomisbf16 xmm22, word ptr [rdx - 256]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x2f,0x72,0x80]
          vcomisbf16 xmm22, word ptr [rdx - 256]

// CHECK: vdivbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5e,0xf0]
          vdivbf16 ymm22, ymm23, ymm24

// CHECK: vdivbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5e,0xf0]
          vdivbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vdivbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5e,0xf0]
          vdivbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vdivbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5e,0xf0]
          vdivbf16 zmm22, zmm23, zmm24

// CHECK: vdivbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5e,0xf0]
          vdivbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vdivbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5e,0xf0]
          vdivbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vdivbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5e,0xf0]
          vdivbf16 xmm22, xmm23, xmm24

// CHECK: vdivbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5e,0xf0]
          vdivbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vdivbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5e,0xf0]
          vdivbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vdivbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vdivbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vdivbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vdivbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vdivbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5e,0x71,0x7f]
          vdivbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vdivbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5e,0x72,0x80]
          vdivbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vdivbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vdivbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vdivbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vdivbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vdivbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5e,0x71,0x7f]
          vdivbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vdivbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5e,0x72,0x80]
          vdivbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vdivbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vdivbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vdivbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5e,0x35,0x00,0x00,0x00,0x00]
          vdivbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vdivbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vdivbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vdivbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5e,0x71,0x7f]
          vdivbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vdivbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5e,0x72,0x80]
          vdivbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd132bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x98,0xf0]
          vfmadd132bf16 ymm22, ymm23, ymm24

// CHECK: vfmadd132bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x98,0xf0]
          vfmadd132bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd132bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x98,0xf0]
          vfmadd132bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd132bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x98,0xf0]
          vfmadd132bf16 zmm22, zmm23, zmm24

// CHECK: vfmadd132bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x98,0xf0]
          vfmadd132bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd132bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x98,0xf0]
          vfmadd132bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd132bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x98,0xf0]
          vfmadd132bf16 xmm22, xmm23, xmm24

// CHECK: vfmadd132bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x98,0xf0]
          vfmadd132bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd132bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x98,0xf0]
          vfmadd132bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x98,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x98,0x71,0x7f]
          vfmadd132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x98,0x72,0x80]
          vfmadd132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x98,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x98,0x71,0x7f]
          vfmadd132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x98,0x72,0x80]
          vfmadd132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x98,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd132bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x98,0x35,0x00,0x00,0x00,0x00]
          vfmadd132bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x98,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x98,0x71,0x7f]
          vfmadd132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x98,0x72,0x80]
          vfmadd132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd213bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xa8,0xf0]
          vfmadd213bf16 ymm22, ymm23, ymm24

// CHECK: vfmadd213bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xa8,0xf0]
          vfmadd213bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd213bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xa8,0xf0]
          vfmadd213bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd213bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xa8,0xf0]
          vfmadd213bf16 zmm22, zmm23, zmm24

// CHECK: vfmadd213bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xa8,0xf0]
          vfmadd213bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd213bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xa8,0xf0]
          vfmadd213bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd213bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xa8,0xf0]
          vfmadd213bf16 xmm22, xmm23, xmm24

// CHECK: vfmadd213bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xa8,0xf0]
          vfmadd213bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd213bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xa8,0xf0]
          vfmadd213bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xa8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xa8,0x71,0x7f]
          vfmadd213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xa8,0x72,0x80]
          vfmadd213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xa8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xa8,0x71,0x7f]
          vfmadd213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xa8,0x72,0x80]
          vfmadd213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xa8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd213bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xa8,0x35,0x00,0x00,0x00,0x00]
          vfmadd213bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xa8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xa8,0x71,0x7f]
          vfmadd213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xa8,0x72,0x80]
          vfmadd213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd231bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xb8,0xf0]
          vfmadd231bf16 ymm22, ymm23, ymm24

// CHECK: vfmadd231bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xb8,0xf0]
          vfmadd231bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmadd231bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xb8,0xf0]
          vfmadd231bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmadd231bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xb8,0xf0]
          vfmadd231bf16 zmm22, zmm23, zmm24

// CHECK: vfmadd231bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xb8,0xf0]
          vfmadd231bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmadd231bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xb8,0xf0]
          vfmadd231bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmadd231bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xb8,0xf0]
          vfmadd231bf16 xmm22, xmm23, xmm24

// CHECK: vfmadd231bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xb8,0xf0]
          vfmadd231bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmadd231bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xb8,0xf0]
          vfmadd231bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmadd231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmadd231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xb8,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmadd231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xb8,0x71,0x7f]
          vfmadd231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmadd231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xb8,0x72,0x80]
          vfmadd231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmadd231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmadd231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xb8,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmadd231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xb8,0x71,0x7f]
          vfmadd231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmadd231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xb8,0x72,0x80]
          vfmadd231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xb8,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmadd231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmadd231bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xb8,0x35,0x00,0x00,0x00,0x00]
          vfmadd231bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmadd231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xb8,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmadd231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xb8,0x71,0x7f]
          vfmadd231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmadd231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xb8,0x72,0x80]
          vfmadd231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub132bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9a,0xf0]
          vfmsub132bf16 ymm22, ymm23, ymm24

// CHECK: vfmsub132bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9a,0xf0]
          vfmsub132bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub132bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9a,0xf0]
          vfmsub132bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub132bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9a,0xf0]
          vfmsub132bf16 zmm22, zmm23, zmm24

// CHECK: vfmsub132bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9a,0xf0]
          vfmsub132bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub132bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9a,0xf0]
          vfmsub132bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub132bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9a,0xf0]
          vfmsub132bf16 xmm22, xmm23, xmm24

// CHECK: vfmsub132bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9a,0xf0]
          vfmsub132bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub132bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9a,0xf0]
          vfmsub132bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9a,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9a,0x71,0x7f]
          vfmsub132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9a,0x72,0x80]
          vfmsub132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9a,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9a,0x71,0x7f]
          vfmsub132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9a,0x72,0x80]
          vfmsub132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9a,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub132bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9a,0x35,0x00,0x00,0x00,0x00]
          vfmsub132bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9a,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9a,0x71,0x7f]
          vfmsub132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9a,0x72,0x80]
          vfmsub132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub213bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xaa,0xf0]
          vfmsub213bf16 ymm22, ymm23, ymm24

// CHECK: vfmsub213bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xaa,0xf0]
          vfmsub213bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub213bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xaa,0xf0]
          vfmsub213bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub213bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xaa,0xf0]
          vfmsub213bf16 zmm22, zmm23, zmm24

// CHECK: vfmsub213bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xaa,0xf0]
          vfmsub213bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub213bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xaa,0xf0]
          vfmsub213bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub213bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xaa,0xf0]
          vfmsub213bf16 xmm22, xmm23, xmm24

// CHECK: vfmsub213bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xaa,0xf0]
          vfmsub213bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub213bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xaa,0xf0]
          vfmsub213bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xaa,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xaa,0x71,0x7f]
          vfmsub213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xaa,0x72,0x80]
          vfmsub213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xaa,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xaa,0x71,0x7f]
          vfmsub213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xaa,0x72,0x80]
          vfmsub213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xaa,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub213bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xaa,0x35,0x00,0x00,0x00,0x00]
          vfmsub213bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xaa,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xaa,0x71,0x7f]
          vfmsub213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xaa,0x72,0x80]
          vfmsub213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub231bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xba,0xf0]
          vfmsub231bf16 ymm22, ymm23, ymm24

// CHECK: vfmsub231bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xba,0xf0]
          vfmsub231bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfmsub231bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xba,0xf0]
          vfmsub231bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfmsub231bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xba,0xf0]
          vfmsub231bf16 zmm22, zmm23, zmm24

// CHECK: vfmsub231bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xba,0xf0]
          vfmsub231bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfmsub231bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xba,0xf0]
          vfmsub231bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfmsub231bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xba,0xf0]
          vfmsub231bf16 xmm22, xmm23, xmm24

// CHECK: vfmsub231bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xba,0xf0]
          vfmsub231bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfmsub231bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xba,0xf0]
          vfmsub231bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfmsub231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfmsub231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xba,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfmsub231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xba,0x71,0x7f]
          vfmsub231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfmsub231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xba,0x72,0x80]
          vfmsub231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfmsub231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfmsub231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xba,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfmsub231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xba,0x71,0x7f]
          vfmsub231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfmsub231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xba,0x72,0x80]
          vfmsub231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xba,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfmsub231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfmsub231bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xba,0x35,0x00,0x00,0x00,0x00]
          vfmsub231bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfmsub231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xba,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfmsub231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xba,0x71,0x7f]
          vfmsub231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfmsub231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xba,0x72,0x80]
          vfmsub231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd132bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9c,0xf0]
          vfnmadd132bf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd132bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9c,0xf0]
          vfnmadd132bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd132bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9c,0xf0]
          vfnmadd132bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd132bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9c,0xf0]
          vfnmadd132bf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd132bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9c,0xf0]
          vfnmadd132bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd132bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9c,0xf0]
          vfnmadd132bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd132bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9c,0xf0]
          vfnmadd132bf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd132bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9c,0xf0]
          vfnmadd132bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd132bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9c,0xf0]
          vfnmadd132bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9c,0x71,0x7f]
          vfnmadd132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9c,0x72,0x80]
          vfnmadd132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9c,0x71,0x7f]
          vfnmadd132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9c,0x72,0x80]
          vfnmadd132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd132bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9c,0x35,0x00,0x00,0x00,0x00]
          vfnmadd132bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9c,0x71,0x7f]
          vfnmadd132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9c,0x72,0x80]
          vfnmadd132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd213bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xac,0xf0]
          vfnmadd213bf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd213bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xac,0xf0]
          vfnmadd213bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd213bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xac,0xf0]
          vfnmadd213bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd213bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xac,0xf0]
          vfnmadd213bf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd213bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xac,0xf0]
          vfnmadd213bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd213bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xac,0xf0]
          vfnmadd213bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd213bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xac,0xf0]
          vfnmadd213bf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd213bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xac,0xf0]
          vfnmadd213bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd213bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xac,0xf0]
          vfnmadd213bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xac,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xac,0x71,0x7f]
          vfnmadd213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xac,0x72,0x80]
          vfnmadd213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xac,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xac,0x71,0x7f]
          vfnmadd213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xac,0x72,0x80]
          vfnmadd213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xac,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd213bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xac,0x35,0x00,0x00,0x00,0x00]
          vfnmadd213bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xac,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xac,0x71,0x7f]
          vfnmadd213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xac,0x72,0x80]
          vfnmadd213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd231bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbc,0xf0]
          vfnmadd231bf16 ymm22, ymm23, ymm24

// CHECK: vfnmadd231bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbc,0xf0]
          vfnmadd231bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmadd231bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbc,0xf0]
          vfnmadd231bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmadd231bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbc,0xf0]
          vfnmadd231bf16 zmm22, zmm23, zmm24

// CHECK: vfnmadd231bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbc,0xf0]
          vfnmadd231bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmadd231bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbc,0xf0]
          vfnmadd231bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmadd231bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbc,0xf0]
          vfnmadd231bf16 xmm22, xmm23, xmm24

// CHECK: vfnmadd231bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbc,0xf0]
          vfnmadd231bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmadd231bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbc,0xf0]
          vfnmadd231bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmadd231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmadd231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbc,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmadd231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbc,0x71,0x7f]
          vfnmadd231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmadd231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbc,0x72,0x80]
          vfnmadd231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmadd231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmadd231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbc,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmadd231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbc,0x71,0x7f]
          vfnmadd231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbc,0x72,0x80]
          vfnmadd231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbc,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmadd231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmadd231bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbc,0x35,0x00,0x00,0x00,0x00]
          vfnmadd231bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmadd231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbc,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmadd231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbc,0x71,0x7f]
          vfnmadd231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbc,0x72,0x80]
          vfnmadd231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub132bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x9e,0xf0]
          vfnmsub132bf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub132bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x9e,0xf0]
          vfnmsub132bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub132bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x9e,0xf0]
          vfnmsub132bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub132bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x9e,0xf0]
          vfnmsub132bf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub132bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x9e,0xf0]
          vfnmsub132bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub132bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x9e,0xf0]
          vfnmsub132bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub132bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x9e,0xf0]
          vfnmsub132bf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub132bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x9e,0xf0]
          vfnmsub132bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub132bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x9e,0xf0]
          vfnmsub132bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x9e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x9e,0x71,0x7f]
          vfnmsub132bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x9e,0x72,0x80]
          vfnmsub132bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x9e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x9e,0x71,0x7f]
          vfnmsub132bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x9e,0x72,0x80]
          vfnmsub132bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x9e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub132bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub132bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x9e,0x35,0x00,0x00,0x00,0x00]
          vfnmsub132bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x9e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x9e,0x71,0x7f]
          vfnmsub132bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x9e,0x72,0x80]
          vfnmsub132bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub213bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xae,0xf0]
          vfnmsub213bf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub213bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xae,0xf0]
          vfnmsub213bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub213bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xae,0xf0]
          vfnmsub213bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub213bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xae,0xf0]
          vfnmsub213bf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub213bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xae,0xf0]
          vfnmsub213bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub213bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xae,0xf0]
          vfnmsub213bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub213bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xae,0xf0]
          vfnmsub213bf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub213bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xae,0xf0]
          vfnmsub213bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub213bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xae,0xf0]
          vfnmsub213bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xae,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xae,0x71,0x7f]
          vfnmsub213bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xae,0x72,0x80]
          vfnmsub213bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xae,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xae,0x71,0x7f]
          vfnmsub213bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xae,0x72,0x80]
          vfnmsub213bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xae,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub213bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub213bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xae,0x35,0x00,0x00,0x00,0x00]
          vfnmsub213bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xae,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xae,0x71,0x7f]
          vfnmsub213bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xae,0x72,0x80]
          vfnmsub213bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub231bf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0xbe,0xf0]
          vfnmsub231bf16 ymm22, ymm23, ymm24

// CHECK: vfnmsub231bf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0xbe,0xf0]
          vfnmsub231bf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vfnmsub231bf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0xbe,0xf0]
          vfnmsub231bf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vfnmsub231bf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0xbe,0xf0]
          vfnmsub231bf16 zmm22, zmm23, zmm24

// CHECK: vfnmsub231bf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0xbe,0xf0]
          vfnmsub231bf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vfnmsub231bf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0xbe,0xf0]
          vfnmsub231bf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vfnmsub231bf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0xbe,0xf0]
          vfnmsub231bf16 xmm22, xmm23, xmm24

// CHECK: vfnmsub231bf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0xbe,0xf0]
          vfnmsub231bf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vfnmsub231bf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0xbe,0xf0]
          vfnmsub231bf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vfnmsub231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231bf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vfnmsub231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0xbe,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231bf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vfnmsub231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0xbe,0x71,0x7f]
          vfnmsub231bf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vfnmsub231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0xbe,0x72,0x80]
          vfnmsub231bf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vfnmsub231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231bf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vfnmsub231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0xbe,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231bf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vfnmsub231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0xbe,0x71,0x7f]
          vfnmsub231bf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0xbe,0x72,0x80]
          vfnmsub231bf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231bf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0xbe,0xb4,0x80,0x23,0x01,0x00,0x00]
          vfnmsub231bf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vfnmsub231bf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0xbe,0x35,0x00,0x00,0x00,0x00]
          vfnmsub231bf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vfnmsub231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0xbe,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231bf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vfnmsub231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0xbe,0x71,0x7f]
          vfnmsub231bf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0xbe,0x72,0x80]
          vfnmsub231bf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vfpclassbf16 k5, zmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x48,0x66,0xef,0x7b]
          vfpclassbf16 k5, zmm23, 123

// CHECK: vfpclassbf16 k5 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x4f,0x66,0xef,0x7b]
          vfpclassbf16 k5 {k7}, zmm23, 123

// CHECK: vfpclassbf16 k5, ymm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x28,0x66,0xef,0x7b]
          vfpclassbf16 k5, ymm23, 123

// CHECK: vfpclassbf16 k5 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x2f,0x66,0xef,0x7b]
          vfpclassbf16 k5 {k7}, ymm23, 123

// CHECK: vfpclassbf16 k5, xmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xef,0x7b]
          vfpclassbf16 k5, xmm23, 123

// CHECK: vfpclassbf16 k5 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x0f,0x66,0xef,0x7b]
          vfpclassbf16 k5 {k7}, xmm23, 123

// CHECK: vfpclassbf16 k5, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x7f,0x08,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclassbf16 k5, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vfpclassbf16 k5 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xd3,0x7f,0x0f,0x66,0xac,0x80,0x23,0x01,0x00,0x00,0x7b]
          vfpclassbf16 k5 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vfpclassbf16 k5, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16 k5, word ptr [rip]{1to8}, 123

// CHECK: vfpclassbf16 k5, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclassbf16 k5, xmmword ptr [2*rbp - 512], 123

// CHECK: vfpclassbf16 k5 {k7}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, xmmword ptr [rcx + 2032], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vfpclassbf16 k5, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16 k5, word ptr [rip]{1to16}, 123

// CHECK: vfpclassbf16 k5, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclassbf16 k5, ymmword ptr [2*rbp - 1024], 123

// CHECK: vfpclassbf16 k5 {k7}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, ymmword ptr [rcx + 4064], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vfpclassbf16 k5, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x2d,0x00,0x00,0x00,0x00,0x7b]
          vfpclassbf16 k5, word ptr [rip]{1to32}, 123

// CHECK: vfpclassbf16 k5, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclassbf16 k5, zmmword ptr [2*rbp - 2048], 123

// CHECK: vfpclassbf16 k5 {k7}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16 k5 {k7}, zmmword ptr [rcx + 8128], 123

// CHECK: vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16 k5 {k7}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vgetexpbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xf7]
          vgetexpbf16 xmm22, xmm23

// CHECK: vgetexpbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x42,0xf7]
          vgetexpbf16 xmm22 {k7}, xmm23

// CHECK: vgetexpbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x42,0xf7]
          vgetexpbf16 xmm22 {k7} {z}, xmm23

// CHECK: vgetexpbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xf7]
          vgetexpbf16 zmm22, zmm23

// CHECK: vgetexpbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x42,0xf7]
          vgetexpbf16 zmm22 {k7}, zmm23

// CHECK: vgetexpbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x42,0xf7]
          vgetexpbf16 zmm22 {k7} {z}, zmm23

// CHECK: vgetexpbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xf7]
          vgetexpbf16 ymm22, ymm23

// CHECK: vgetexpbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x42,0xf7]
          vgetexpbf16 ymm22 {k7}, ymm23

// CHECK: vgetexpbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x42,0xf7]
          vgetexpbf16 ymm22 {k7} {z}, ymm23

// CHECK: vgetexpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexpbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vgetexpbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x42,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vgetexpbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vgetexpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x42,0x71,0x7f]
          vgetexpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vgetexpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x42,0x72,0x80]
          vgetexpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vgetexpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexpbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vgetexpbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x42,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vgetexpbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vgetexpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x42,0x71,0x7f]
          vgetexpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vgetexpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x42,0x72,0x80]
          vgetexpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vgetexpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x42,0xb4,0x80,0x23,0x01,0x00,0x00]
          vgetexpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vgetexpbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x42,0x35,0x00,0x00,0x00,0x00]
          vgetexpbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vgetexpbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x42,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vgetexpbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vgetexpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x42,0x71,0x7f]
          vgetexpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vgetexpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x42,0x72,0x80]
          vgetexpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vgetmantbf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xf7,0x7b]
          vgetmantbf16 zmm22, zmm23, 123

// CHECK: vgetmantbf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x26,0xf7,0x7b]
          vgetmantbf16 zmm22 {k7}, zmm23, 123

// CHECK: vgetmantbf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x26,0xf7,0x7b]
          vgetmantbf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vgetmantbf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xf7,0x7b]
          vgetmantbf16 ymm22, ymm23, 123

// CHECK: vgetmantbf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x26,0xf7,0x7b]
          vgetmantbf16 ymm22 {k7}, ymm23, 123

// CHECK: vgetmantbf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x26,0xf7,0x7b]
          vgetmantbf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vgetmantbf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xf7,0x7b]
          vgetmantbf16 xmm22, xmm23, 123

// CHECK: vgetmantbf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x26,0xf7,0x7b]
          vgetmantbf16 xmm22 {k7}, xmm23, 123

// CHECK: vgetmantbf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x26,0xf7,0x7b]
          vgetmantbf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vgetmantbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantbf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vgetmantbf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x26,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantbf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vgetmantbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x26,0x71,0x7f,0x7b]
          vgetmantbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vgetmantbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vgetmantbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantbf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vgetmantbf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x26,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantbf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vgetmantbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x26,0x71,0x7f,0x7b]
          vgetmantbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vgetmantbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vgetmantbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x26,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vgetmantbf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x26,0x35,0x00,0x00,0x00,0x00,0x7b]
          vgetmantbf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vgetmantbf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x26,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantbf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vgetmantbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x26,0x71,0x7f,0x7b]
          vgetmantbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vgetmantbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vmaxbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5f,0xf0]
          vmaxbf16 ymm22, ymm23, ymm24

// CHECK: vmaxbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5f,0xf0]
          vmaxbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vmaxbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5f,0xf0]
          vmaxbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vmaxbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5f,0xf0]
          vmaxbf16 zmm22, zmm23, zmm24

// CHECK: vmaxbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5f,0xf0]
          vmaxbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vmaxbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5f,0xf0]
          vmaxbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vmaxbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5f,0xf0]
          vmaxbf16 xmm22, xmm23, xmm24

// CHECK: vmaxbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5f,0xf0]
          vmaxbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vmaxbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5f,0xf0]
          vmaxbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vmaxbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vmaxbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5f,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmaxbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vmaxbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5f,0x71,0x7f]
          vmaxbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vmaxbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5f,0x72,0x80]
          vmaxbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vmaxbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vmaxbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5f,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmaxbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vmaxbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5f,0x71,0x7f]
          vmaxbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vmaxbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5f,0x72,0x80]
          vmaxbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vmaxbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5f,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmaxbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vmaxbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5f,0x35,0x00,0x00,0x00,0x00]
          vmaxbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vmaxbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5f,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmaxbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vmaxbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5f,0x71,0x7f]
          vmaxbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vmaxbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5f,0x72,0x80]
          vmaxbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vminbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5d,0xf0]
          vminbf16 ymm22, ymm23, ymm24

// CHECK: vminbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5d,0xf0]
          vminbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vminbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5d,0xf0]
          vminbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vminbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5d,0xf0]
          vminbf16 zmm22, zmm23, zmm24

// CHECK: vminbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5d,0xf0]
          vminbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vminbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5d,0xf0]
          vminbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vminbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5d,0xf0]
          vminbf16 xmm22, xmm23, xmm24

// CHECK: vminbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5d,0xf0]
          vminbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vminbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5d,0xf0]
          vminbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vminbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vminbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vminbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vminbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vminbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5d,0x71,0x7f]
          vminbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vminbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5d,0x72,0x80]
          vminbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vminbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vminbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vminbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vminbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vminbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5d,0x71,0x7f]
          vminbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vminbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5d,0x72,0x80]
          vminbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vminbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vminbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vminbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5d,0x35,0x00,0x00,0x00,0x00]
          vminbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vminbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vminbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vminbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5d,0x71,0x7f]
          vminbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vminbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5d,0x72,0x80]
          vminbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vmulbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x59,0xf0]
          vmulbf16 ymm22, ymm23, ymm24

// CHECK: vmulbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x59,0xf0]
          vmulbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vmulbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x59,0xf0]
          vmulbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vmulbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x59,0xf0]
          vmulbf16 zmm22, zmm23, zmm24

// CHECK: vmulbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x59,0xf0]
          vmulbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vmulbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x59,0xf0]
          vmulbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vmulbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x59,0xf0]
          vmulbf16 xmm22, xmm23, xmm24

// CHECK: vmulbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x59,0xf0]
          vmulbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vmulbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x59,0xf0]
          vmulbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vmulbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vmulbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vmulbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x59,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vmulbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vmulbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x59,0x71,0x7f]
          vmulbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vmulbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x59,0x72,0x80]
          vmulbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vmulbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vmulbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vmulbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x59,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vmulbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vmulbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x59,0x71,0x7f]
          vmulbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vmulbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x59,0x72,0x80]
          vmulbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vmulbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x59,0xb4,0x80,0x23,0x01,0x00,0x00]
          vmulbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vmulbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x59,0x35,0x00,0x00,0x00,0x00]
          vmulbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vmulbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x59,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vmulbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vmulbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x59,0x71,0x7f]
          vmulbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vmulbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x59,0x72,0x80]
          vmulbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vrcpbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xf7]
          vrcpbf16 xmm22, xmm23

// CHECK: vrcpbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4c,0xf7]
          vrcpbf16 xmm22 {k7}, xmm23

// CHECK: vrcpbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4c,0xf7]
          vrcpbf16 xmm22 {k7} {z}, xmm23

// CHECK: vrcpbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xf7]
          vrcpbf16 zmm22, zmm23

// CHECK: vrcpbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4c,0xf7]
          vrcpbf16 zmm22 {k7}, zmm23

// CHECK: vrcpbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4c,0xf7]
          vrcpbf16 zmm22 {k7} {z}, zmm23

// CHECK: vrcpbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xf7]
          vrcpbf16 ymm22, ymm23

// CHECK: vrcpbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4c,0xf7]
          vrcpbf16 ymm22 {k7}, ymm23

// CHECK: vrcpbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4c,0xf7]
          vrcpbf16 ymm22 {k7} {z}, ymm23

// CHECK: vrcpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vrcpbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vrcpbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrcpbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vrcpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4c,0x71,0x7f]
          vrcpbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vrcpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4c,0x72,0x80]
          vrcpbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrcpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vrcpbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vrcpbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrcpbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vrcpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4c,0x71,0x7f]
          vrcpbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vrcpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4c,0x72,0x80]
          vrcpbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vrcpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrcpbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vrcpbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4c,0x35,0x00,0x00,0x00,0x00]
          vrcpbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vrcpbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrcpbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vrcpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4c,0x71,0x7f]
          vrcpbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vrcpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4c,0x72,0x80]
          vrcpbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vreducebf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xf7,0x7b]
          vreducebf16 zmm22, zmm23, 123

// CHECK: vreducebf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x56,0xf7,0x7b]
          vreducebf16 zmm22 {k7}, zmm23, 123

// CHECK: vreducebf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x56,0xf7,0x7b]
          vreducebf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vreducebf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xf7,0x7b]
          vreducebf16 ymm22, ymm23, 123

// CHECK: vreducebf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x56,0xf7,0x7b]
          vreducebf16 ymm22 {k7}, ymm23, 123

// CHECK: vreducebf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x56,0xf7,0x7b]
          vreducebf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vreducebf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xf7,0x7b]
          vreducebf16 xmm22, xmm23, 123

// CHECK: vreducebf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x56,0xf7,0x7b]
          vreducebf16 xmm22 {k7}, xmm23, 123

// CHECK: vreducebf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x56,0xf7,0x7b]
          vreducebf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vreducebf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducebf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducebf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vreducebf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x56,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducebf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vreducebf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x56,0x71,0x7f,0x7b]
          vreducebf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vreducebf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x56,0x72,0x80,0x7b]
          vreducebf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vreducebf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducebf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducebf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vreducebf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x56,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducebf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vreducebf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x56,0x71,0x7f,0x7b]
          vreducebf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vreducebf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x56,0x72,0x80,0x7b]
          vreducebf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vreducebf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreducebf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x56,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vreducebf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x56,0x35,0x00,0x00,0x00,0x00,0x7b]
          vreducebf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vreducebf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x56,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducebf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vreducebf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x56,0x71,0x7f,0x7b]
          vreducebf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vreducebf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x56,0x72,0x80,0x7b]
          vreducebf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vrndscalebf16 zmm22, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xf7,0x7b]
          vrndscalebf16 zmm22, zmm23, 123

// CHECK: vrndscalebf16 zmm22 {k7}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x4f,0x08,0xf7,0x7b]
          vrndscalebf16 zmm22 {k7}, zmm23, 123

// CHECK: vrndscalebf16 zmm22 {k7} {z}, zmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xcf,0x08,0xf7,0x7b]
          vrndscalebf16 zmm22 {k7} {z}, zmm23, 123

// CHECK: vrndscalebf16 ymm22, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xf7,0x7b]
          vrndscalebf16 ymm22, ymm23, 123

// CHECK: vrndscalebf16 ymm22 {k7}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x2f,0x08,0xf7,0x7b]
          vrndscalebf16 ymm22 {k7}, ymm23, 123

// CHECK: vrndscalebf16 ymm22 {k7} {z}, ymm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0xaf,0x08,0xf7,0x7b]
          vrndscalebf16 ymm22 {k7} {z}, ymm23, 123

// CHECK: vrndscalebf16 xmm22, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xf7,0x7b]
          vrndscalebf16 xmm22, xmm23, 123

// CHECK: vrndscalebf16 xmm22 {k7}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x0f,0x08,0xf7,0x7b]
          vrndscalebf16 xmm22 {k7}, xmm23, 123

// CHECK: vrndscalebf16 xmm22 {k7} {z}, xmm23, 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x8f,0x08,0xf7,0x7b]
          vrndscalebf16 xmm22 {k7} {z}, xmm23, 123

// CHECK: vrndscalebf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x08,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalebf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x0f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalebf16 xmm22, word ptr [rip]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x18,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16 xmm22, word ptr [rip]{1to8}, 123

// CHECK: vrndscalebf16 xmm22, xmmword ptr [2*rbp - 512], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x08,0x08,0x34,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalebf16 xmm22, xmmword ptr [2*rbp - 512], 123

// CHECK: vrndscalebf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x8f,0x08,0x71,0x7f,0x7b]
          vrndscalebf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032], 123

// CHECK: vrndscalebf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x9f,0x08,0x72,0x80,0x7b]
          vrndscalebf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vrndscalebf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x28,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalebf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x2f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalebf16 ymm22, word ptr [rip]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x38,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16 ymm22, word ptr [rip]{1to16}, 123

// CHECK: vrndscalebf16 ymm22, ymmword ptr [2*rbp - 1024], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x28,0x08,0x34,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalebf16 ymm22, ymmword ptr [2*rbp - 1024], 123

// CHECK: vrndscalebf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xaf,0x08,0x71,0x7f,0x7b]
          vrndscalebf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064], 123

// CHECK: vrndscalebf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xbf,0x08,0x72,0x80,0x7b]
          vrndscalebf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vrndscalebf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xa3,0x7f,0x48,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscalebf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123
// CHECK: encoding: [0x62,0xc3,0x7f,0x4f,0x08,0xb4,0x80,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291], 123

// CHECK: vrndscalebf16 zmm22, word ptr [rip]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x58,0x08,0x35,0x00,0x00,0x00,0x00,0x7b]
          vrndscalebf16 zmm22, word ptr [rip]{1to32}, 123

// CHECK: vrndscalebf16 zmm22, zmmword ptr [2*rbp - 2048], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x08,0x34,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalebf16 zmm22, zmmword ptr [2*rbp - 2048], 123

// CHECK: vrndscalebf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xcf,0x08,0x71,0x7f,0x7b]
          vrndscalebf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128], 123

// CHECK: vrndscalebf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0xdf,0x08,0x72,0x80,0x7b]
          vrndscalebf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}, 123

// CHECK: vrsqrtbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xf7]
          vrsqrtbf16 xmm22, xmm23

// CHECK: vrsqrtbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x0f,0x4e,0xf7]
          vrsqrtbf16 xmm22 {k7}, xmm23

// CHECK: vrsqrtbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x8f,0x4e,0xf7]
          vrsqrtbf16 xmm22 {k7} {z}, xmm23

// CHECK: vrsqrtbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xf7]
          vrsqrtbf16 zmm22, zmm23

// CHECK: vrsqrtbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x4f,0x4e,0xf7]
          vrsqrtbf16 zmm22 {k7}, zmm23

// CHECK: vrsqrtbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xcf,0x4e,0xf7]
          vrsqrtbf16 zmm22 {k7} {z}, zmm23

// CHECK: vrsqrtbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xf7]
          vrsqrtbf16 ymm22, ymm23

// CHECK: vrsqrtbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0x2f,0x4e,0xf7]
          vrsqrtbf16 ymm22 {k7}, ymm23

// CHECK: vrsqrtbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa6,0x7c,0xaf,0x4e,0xf7]
          vrsqrtbf16 ymm22 {k7} {z}, ymm23

// CHECK: vrsqrtbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x08,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x0f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x18,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vrsqrtbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x7c,0x08,0x4e,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vrsqrtbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x7c,0x8f,0x4e,0x71,0x7f]
          vrsqrtbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vrsqrtbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x7c,0x9f,0x4e,0x72,0x80]
          vrsqrtbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrsqrtbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x28,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x2f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0x38,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vrsqrtbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x7c,0x28,0x4e,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vrsqrtbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x7c,0xaf,0x4e,0x71,0x7f]
          vrsqrtbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vrsqrtbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x7c,0xbf,0x4e,0x72,0x80]
          vrsqrtbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vrsqrtbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x7c,0x48,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x7c,0x4f,0x4e,0xb4,0x80,0x23,0x01,0x00,0x00]
          vrsqrtbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vrsqrtbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0x58,0x4e,0x35,0x00,0x00,0x00,0x00]
          vrsqrtbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vrsqrtbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x7c,0x48,0x4e,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vrsqrtbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x7c,0xcf,0x4e,0x71,0x7f]
          vrsqrtbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vrsqrtbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x7c,0xdf,0x4e,0x72,0x80]
          vrsqrtbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vscalefbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x20,0x2c,0xf0]
          vscalefbf16 ymm22, ymm23, ymm24

// CHECK: vscalefbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0x27,0x2c,0xf0]
          vscalefbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vscalefbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x86,0x44,0xa7,0x2c,0xf0]
          vscalefbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vscalefbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x40,0x2c,0xf0]
          vscalefbf16 zmm22, zmm23, zmm24

// CHECK: vscalefbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0x47,0x2c,0xf0]
          vscalefbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vscalefbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x86,0x44,0xc7,0x2c,0xf0]
          vscalefbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vscalefbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x00,0x2c,0xf0]
          vscalefbf16 xmm22, xmm23, xmm24

// CHECK: vscalefbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x07,0x2c,0xf0]
          vscalefbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vscalefbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x86,0x44,0x87,0x2c,0xf0]
          vscalefbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vscalefbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x40,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x47,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0x50,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vscalefbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe6,0x44,0x40,0x2c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vscalefbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vscalefbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe6,0x44,0xc7,0x2c,0x71,0x7f]
          vscalefbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vscalefbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe6,0x44,0xd7,0x2c,0x72,0x80]
          vscalefbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vscalefbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x20,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x27,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0x30,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vscalefbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe6,0x44,0x20,0x2c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vscalefbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vscalefbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe6,0x44,0xa7,0x2c,0x71,0x7f]
          vscalefbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vscalefbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe6,0x44,0xb7,0x2c,0x72,0x80]
          vscalefbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vscalefbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa6,0x44,0x00,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc6,0x44,0x07,0x2c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vscalefbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vscalefbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x10,0x2c,0x35,0x00,0x00,0x00,0x00]
          vscalefbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vscalefbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe6,0x44,0x00,0x2c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vscalefbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vscalefbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe6,0x44,0x87,0x2c,0x71,0x7f]
          vscalefbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vscalefbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe6,0x44,0x97,0x2c,0x72,0x80]
          vscalefbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtbf16 xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xf7]
          vsqrtbf16 xmm22, xmm23

// CHECK: vsqrtbf16 xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x51,0xf7]
          vsqrtbf16 xmm22 {k7}, xmm23

// CHECK: vsqrtbf16 xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x51,0xf7]
          vsqrtbf16 xmm22 {k7} {z}, xmm23

// CHECK: vsqrtbf16 zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xf7]
          vsqrtbf16 zmm22, zmm23

// CHECK: vsqrtbf16 zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x51,0xf7]
          vsqrtbf16 zmm22 {k7}, zmm23

// CHECK: vsqrtbf16 zmm22 {k7} {z}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xcf,0x51,0xf7]
          vsqrtbf16 zmm22 {k7} {z}, zmm23

// CHECK: vsqrtbf16 ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xf7]
          vsqrtbf16 ymm22, ymm23

// CHECK: vsqrtbf16 ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x51,0xf7]
          vsqrtbf16 ymm22 {k7}, ymm23

// CHECK: vsqrtbf16 ymm22 {k7} {z}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0xaf,0x51,0xf7]
          vsqrtbf16 ymm22 {k7} {z}, ymm23

// CHECK: vsqrtbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16 xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16 xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtbf16 xmm22, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16 xmm22, word ptr [rip]{1to8}

// CHECK: vsqrtbf16 xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtbf16 xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vsqrtbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x51,0x71,0x7f]
          vsqrtbf16 xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vsqrtbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x51,0x72,0x80]
          vsqrtbf16 xmm22 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16 ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16 ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtbf16 ymm22, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16 ymm22, word ptr [rip]{1to16}

// CHECK: vsqrtbf16 ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtbf16 ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vsqrtbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x51,0x71,0x7f]
          vsqrtbf16 ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vsqrtbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x51,0x72,0x80]
          vsqrtbf16 ymm22 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vsqrtbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtbf16 zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x51,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsqrtbf16 zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vsqrtbf16 zmm22, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x51,0x35,0x00,0x00,0x00,0x00]
          vsqrtbf16 zmm22, word ptr [rip]{1to32}

// CHECK: vsqrtbf16 zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x51,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtbf16 zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vsqrtbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x51,0x71,0x7f]
          vsqrtbf16 zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vsqrtbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x51,0x72,0x80]
          vsqrtbf16 zmm22 {k7} {z}, word ptr [rdx - 256]{1to32}

// CHECK: vsubbf16 ymm22, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x20,0x5c,0xf0]
          vsubbf16 ymm22, ymm23, ymm24

// CHECK: vsubbf16 ymm22 {k7}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0x27,0x5c,0xf0]
          vsubbf16 ymm22 {k7}, ymm23, ymm24

// CHECK: vsubbf16 ymm22 {k7} {z}, ymm23, ymm24
// CHECK: encoding: [0x62,0x85,0x45,0xa7,0x5c,0xf0]
          vsubbf16 ymm22 {k7} {z}, ymm23, ymm24

// CHECK: vsubbf16 zmm22, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x40,0x5c,0xf0]
          vsubbf16 zmm22, zmm23, zmm24

// CHECK: vsubbf16 zmm22 {k7}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0x47,0x5c,0xf0]
          vsubbf16 zmm22 {k7}, zmm23, zmm24

// CHECK: vsubbf16 zmm22 {k7} {z}, zmm23, zmm24
// CHECK: encoding: [0x62,0x85,0x45,0xc7,0x5c,0xf0]
          vsubbf16 zmm22 {k7} {z}, zmm23, zmm24

// CHECK: vsubbf16 xmm22, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x00,0x5c,0xf0]
          vsubbf16 xmm22, xmm23, xmm24

// CHECK: vsubbf16 xmm22 {k7}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x07,0x5c,0xf0]
          vsubbf16 xmm22 {k7}, xmm23, xmm24

// CHECK: vsubbf16 xmm22 {k7} {z}, xmm23, xmm24
// CHECK: encoding: [0x62,0x85,0x45,0x87,0x5c,0xf0]
          vsubbf16 xmm22 {k7} {z}, xmm23, xmm24

// CHECK: vsubbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x40,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16 zmm22, zmm23, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x47,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16 zmm22 {k7}, zmm23, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vsubbf16 zmm22, zmm23, word ptr [rip]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0x50,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16 zmm22, zmm23, word ptr [rip]{1to32}

// CHECK: vsubbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x45,0x40,0x5c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vsubbf16 zmm22, zmm23, zmmword ptr [2*rbp - 2048]

// CHECK: vsubbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x45,0xc7,0x5c,0x71,0x7f]
          vsubbf16 zmm22 {k7} {z}, zmm23, zmmword ptr [rcx + 8128]

// CHECK: vsubbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}
// CHECK: encoding: [0x62,0xe5,0x45,0xd7,0x5c,0x72,0x80]
          vsubbf16 zmm22 {k7} {z}, zmm23, word ptr [rdx - 256]{1to32}

// CHECK: vsubbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x20,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16 ymm22, ymm23, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x27,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16 ymm22 {k7}, ymm23, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vsubbf16 ymm22, ymm23, word ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0x30,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16 ymm22, ymm23, word ptr [rip]{1to16}

// CHECK: vsubbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x45,0x20,0x5c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vsubbf16 ymm22, ymm23, ymmword ptr [2*rbp - 1024]

// CHECK: vsubbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x45,0xa7,0x5c,0x71,0x7f]
          vsubbf16 ymm22 {k7} {z}, ymm23, ymmword ptr [rcx + 4064]

// CHECK: vsubbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xe5,0x45,0xb7,0x5c,0x72,0x80]
          vsubbf16 ymm22 {k7} {z}, ymm23, word ptr [rdx - 256]{1to16}

// CHECK: vsubbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x45,0x00,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubbf16 xmm22, xmm23, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x45,0x07,0x5c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vsubbf16 xmm22 {k7}, xmm23, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vsubbf16 xmm22, xmm23, word ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x10,0x5c,0x35,0x00,0x00,0x00,0x00]
          vsubbf16 xmm22, xmm23, word ptr [rip]{1to8}

// CHECK: vsubbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x45,0x00,0x5c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vsubbf16 xmm22, xmm23, xmmword ptr [2*rbp - 512]

// CHECK: vsubbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x45,0x87,0x5c,0x71,0x7f]
          vsubbf16 xmm22 {k7} {z}, xmm23, xmmword ptr [rcx + 2032]

// CHECK: vsubbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xe5,0x45,0x97,0x5c,0x72,0x80]
          vsubbf16 xmm22 {k7} {z}, xmm23, word ptr [rdx - 256]{1to8}

