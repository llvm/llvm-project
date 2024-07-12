# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: vpermb	xmm30 {k7}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x07,0x8d,0xf4]
         vpermb	xmm30 {k7}, xmm29, xmm28

# CHECK: vpermb	xmm30 {k7} {z}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x87,0x8d,0xf4]
         vpermb	xmm30 {k7} {z}, xmm29, xmm28

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x8d,0x31]
         vpermb	xmm30, xmm29, xmmword ptr [rcx]

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x8d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rdx + 2032]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x8d,0x72,0x7f]
         vpermb	xmm30, xmm29, xmmword ptr [rdx + 2032]

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rdx + 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x8d,0xb2,0x00,0x08,0x00,0x00]
         vpermb	xmm30, xmm29, xmmword ptr [rdx + 2048]

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rdx - 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x8d,0x72,0x80]
         vpermb	xmm30, xmm29, xmmword ptr [rdx - 2048]

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rdx - 2064]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x8d,0xb2,0xf0,0xf7,0xff,0xff]
         vpermb	xmm30, xmm29, xmmword ptr [rdx - 2064]

# CHECK: vpermb	ymm30, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x20,0x8d,0xf4]
         vpermb	ymm30, ymm29, ymm28

# CHECK: vpermb	ymm30 {k7}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x27,0x8d,0xf4]
         vpermb	ymm30 {k7}, ymm29, ymm28

# CHECK: vpermb	ymm30 {k7} {z}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0xa7,0x8d,0xf4]
         vpermb	ymm30 {k7} {z}, ymm29, ymm28

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x8d,0x31]
         vpermb	ymm30, ymm29, ymmword ptr [rcx]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x8d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rdx + 4064]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x8d,0x72,0x7f]
         vpermb	ymm30, ymm29, ymmword ptr [rdx + 4064]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rdx + 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x8d,0xb2,0x00,0x10,0x00,0x00]
         vpermb	ymm30, ymm29, ymmword ptr [rdx + 4096]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rdx - 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x8d,0x72,0x80]
         vpermb	ymm30, ymm29, ymmword ptr [rdx - 4096]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rdx - 4128]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x8d,0xb2,0xe0,0xef,0xff,0xff]
         vpermb	ymm30, ymm29, ymmword ptr [rdx - 4128]

# CHECK: vpermb	xmm30, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x00,0x8d,0xf4]
         vpermb	xmm30, xmm29, xmm28

# CHECK: vpermb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x8d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x8d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermb	zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x40,0x8d,0xf4]
         vpermb	zmm30, zmm29, zmm28

# CHECK: vpermb	zmm30 {k7}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x47,0x8d,0xf4]
         vpermb	zmm30 {k7}, zmm29, zmm28

# CHECK: vpermb	zmm30 {k7} {z}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0xc7,0x8d,0xf4]
         vpermb	zmm30 {k7} {z}, zmm29, zmm28

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x8d,0x31]
         vpermb	zmm30, zmm29, zmmword ptr [rcx]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x8d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rdx + 8128]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x8d,0x72,0x7f]
         vpermb	zmm30, zmm29, zmmword ptr [rdx + 8128]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rdx + 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x8d,0xb2,0x00,0x20,0x00,0x00]
         vpermb	zmm30, zmm29, zmmword ptr [rdx + 8192]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rdx - 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x8d,0x72,0x80]
         vpermb	zmm30, zmm29, zmmword ptr [rdx - 8192]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rdx - 8256]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x8d,0xb2,0xc0,0xdf,0xff,0xff]
         vpermb	zmm30, zmm29, zmmword ptr [rdx - 8256]

# CHECK: vpermb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x8d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermt2b	xmm30, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x00,0x7d,0xf4]
         vpermt2b	xmm30, xmm29, xmm28

# CHECK: vpermt2b	xmm30 {k7}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x07,0x7d,0xf4]
         vpermt2b	xmm30 {k7}, xmm29, xmm28

# CHECK: vpermt2b	xmm30 {k7} {z}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x87,0x7d,0xf4]
         vpermt2b	xmm30 {k7} {z}, xmm29, xmm28

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x7d,0x31]
         vpermt2b	xmm30, xmm29, xmmword ptr [rcx]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x7d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermt2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rdx + 2032]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x7d,0x72,0x7f]
         vpermt2b	xmm30, xmm29, xmmword ptr [rdx + 2032]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rdx + 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x7d,0xb2,0x00,0x08,0x00,0x00]
         vpermt2b	xmm30, xmm29, xmmword ptr [rdx + 2048]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rdx - 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x7d,0x72,0x80]
         vpermt2b	xmm30, xmm29, xmmword ptr [rdx - 2048]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rdx - 2064]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x7d,0xb2,0xf0,0xf7,0xff,0xff]
         vpermt2b	xmm30, xmm29, xmmword ptr [rdx - 2064]

# CHECK: vpermt2b	ymm30, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x20,0x7d,0xf4]
         vpermt2b	ymm30, ymm29, ymm28

# CHECK: vpermt2b	ymm30 {k7}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x27,0x7d,0xf4]
         vpermt2b	ymm30 {k7}, ymm29, ymm28

# CHECK: vpermt2b	ymm30 {k7} {z}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0xa7,0x7d,0xf4]
         vpermt2b	ymm30 {k7} {z}, ymm29, ymm28

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x7d,0x31]
         vpermt2b	ymm30, ymm29, ymmword ptr [rcx]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x7d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermt2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rdx + 4064]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x7d,0x72,0x7f]
         vpermt2b	ymm30, ymm29, ymmword ptr [rdx + 4064]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rdx + 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x7d,0xb2,0x00,0x10,0x00,0x00]
         vpermt2b	ymm30, ymm29, ymmword ptr [rdx + 4096]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rdx - 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x7d,0x72,0x80]
         vpermt2b	ymm30, ymm29, ymmword ptr [rdx - 4096]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rdx - 4128]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x7d,0xb2,0xe0,0xef,0xff,0xff]
         vpermt2b	ymm30, ymm29, ymmword ptr [rdx - 4128]

# CHECK: vpermt2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x7d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermt2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermt2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x7d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermt2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermt2b	zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x40,0x7d,0xf4]
         vpermt2b	zmm30, zmm29, zmm28

# CHECK: vpermt2b	zmm30 {k7}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x47,0x7d,0xf4]
         vpermt2b	zmm30 {k7}, zmm29, zmm28

# CHECK: vpermt2b	zmm30 {k7} {z}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0xc7,0x7d,0xf4]
         vpermt2b	zmm30 {k7} {z}, zmm29, zmm28

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x7d,0x31]
         vpermt2b	zmm30, zmm29, zmmword ptr [rcx]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x7d,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermt2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rdx + 8128]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x7d,0x72,0x7f]
         vpermt2b	zmm30, zmm29, zmmword ptr [rdx + 8128]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rdx + 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x7d,0xb2,0x00,0x20,0x00,0x00]
         vpermt2b	zmm30, zmm29, zmmword ptr [rdx + 8192]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rdx - 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x7d,0x72,0x80]
         vpermt2b	zmm30, zmm29, zmmword ptr [rdx - 8192]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rdx - 8256]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x7d,0xb2,0xc0,0xdf,0xff,0xff]
         vpermt2b	zmm30, zmm29, zmmword ptr [rdx - 8256]

# CHECK: vpermt2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x7d,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermt2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermi2b	xmm30, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x00,0x75,0xf4]
         vpermi2b	xmm30, xmm29, xmm28

# CHECK: vpermi2b	xmm30 {k7}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x07,0x75,0xf4]
         vpermi2b	xmm30 {k7}, xmm29, xmm28

# CHECK: vpermi2b	xmm30 {k7} {z}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x15,0x87,0x75,0xf4]
         vpermi2b	xmm30 {k7} {z}, xmm29, xmm28

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x75,0x31]
         vpermi2b	xmm30, xmm29, xmmword ptr [rcx]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x75,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermi2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rdx + 2032]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x75,0x72,0x7f]
         vpermi2b	xmm30, xmm29, xmmword ptr [rdx + 2032]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rdx + 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x75,0xb2,0x00,0x08,0x00,0x00]
         vpermi2b	xmm30, xmm29, xmmword ptr [rdx + 2048]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rdx - 2048]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x75,0x72,0x80]
         vpermi2b	xmm30, xmm29, xmmword ptr [rdx - 2048]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rdx - 2064]
# CHECK: encoding: [0x62,0x62,0x15,0x00,0x75,0xb2,0xf0,0xf7,0xff,0xff]
         vpermi2b	xmm30, xmm29, xmmword ptr [rdx - 2064]

# CHECK: vpermi2b	ymm30, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x20,0x75,0xf4]
         vpermi2b	ymm30, ymm29, ymm28

# CHECK: vpermi2b	ymm30 {k7}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0x27,0x75,0xf4]
         vpermi2b	ymm30 {k7}, ymm29, ymm28

# CHECK: vpermi2b	ymm30 {k7} {z}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x15,0xa7,0x75,0xf4]
         vpermi2b	ymm30 {k7} {z}, ymm29, ymm28

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x75,0x31]
         vpermi2b	ymm30, ymm29, ymmword ptr [rcx]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x75,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermi2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rdx + 4064]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x75,0x72,0x7f]
         vpermi2b	ymm30, ymm29, ymmword ptr [rdx + 4064]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rdx + 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x75,0xb2,0x00,0x10,0x00,0x00]
         vpermi2b	ymm30, ymm29, ymmword ptr [rdx + 4096]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rdx - 4096]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x75,0x72,0x80]
         vpermi2b	ymm30, ymm29, ymmword ptr [rdx - 4096]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rdx - 4128]
# CHECK: encoding: [0x62,0x62,0x15,0x20,0x75,0xb2,0xe0,0xef,0xff,0xff]
         vpermi2b	ymm30, ymm29, ymmword ptr [rdx - 4128]

# CHECK: vpermi2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x00,0x75,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermi2b	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermi2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x20,0x75,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermi2b	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]

# CHECK: vpermi2b	zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x40,0x75,0xf4]
         vpermi2b	zmm30, zmm29, zmm28

# CHECK: vpermi2b	zmm30 {k7}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0x47,0x75,0xf4]
         vpermi2b	zmm30 {k7}, zmm29, zmm28

# CHECK: vpermi2b	zmm30 {k7} {z}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x15,0xc7,0x75,0xf4]
         vpermi2b	zmm30 {k7} {z}, zmm29, zmm28

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x75,0x31]
         vpermi2b	zmm30, zmm29, zmmword ptr [rcx]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x75,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpermi2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rdx + 8128]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x75,0x72,0x7f]
         vpermi2b	zmm30, zmm29, zmmword ptr [rdx + 8128]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rdx + 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x75,0xb2,0x00,0x20,0x00,0x00]
         vpermi2b	zmm30, zmm29, zmmword ptr [rdx + 8192]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rdx - 8192]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x75,0x72,0x80]
         vpermi2b	zmm30, zmm29, zmmword ptr [rdx - 8192]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rdx - 8256]
# CHECK: encoding: [0x62,0x62,0x15,0x40,0x75,0xb2,0xc0,0xdf,0xff,0xff]
         vpermi2b	zmm30, zmm29, zmmword ptr [rdx - 8256]

# CHECK: vpermi2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x15,0x40,0x75,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpermi2b	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpmultishiftqb	xmm30, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x95,0x00,0x83,0xf4]
         vpmultishiftqb	xmm30, xmm29, xmm28

# CHECK: vpmultishiftqb	xmm30 {k7}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x95,0x07,0x83,0xf4]
         vpmultishiftqb	xmm30 {k7}, xmm29, xmm28

# CHECK: vpmultishiftqb	xmm30 {k7} {z}, xmm29, xmm28
# CHECK: encoding: [0x62,0x02,0x95,0x87,0x83,0xf4]
         vpmultishiftqb	xmm30 {k7} {z}, xmm29, xmm28

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x95,0x00,0x83,0x31]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rcx]

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x95,0x00,0x83,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 291]

# CHECK: vpmultishiftqb	xmm30, xmm29, qword ptr [rcx]{1to2}
# CHECK: encoding: [0x62,0x62,0x95,0x10,0x83,0x31]
         vpmultishiftqb	xmm30, xmm29, qword ptr [rcx]{1to2}


# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx + 2032]
# CHECK: encoding: [0x62,0x62,0x95,0x00,0x83,0x72,0x7f]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx + 2032]

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx + 2048]
# CHECK: encoding: [0x62,0x62,0x95,0x00,0x83,0xb2,0x00,0x08,0x00,0x00]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx + 2048]

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx - 2048]
# CHECK: encoding: [0x62,0x62,0x95,0x00,0x83,0x72,0x80]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx - 2048]

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx - 2064]
# CHECK: encoding: [0x62,0x62,0x95,0x00,0x83,0xb2,0xf0,0xf7,0xff,0xff]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rdx - 2064]

# CHECK: vpmultishiftqb	xmm30, xmm29, qword ptr [rdx + 1016]{1to2}
# CHECK: encoding: [0x62,0x62,0x95,0x10,0x83,0x72,0x7f]
         vpmultishiftqb	xmm30, xmm29, qword ptr [rdx + 1016]{1to2}

# CHECK: vpmultishiftqb	xmm30, xmm29, qword ptr [rdx + 1024]{1to2}
# CHECK: encoding: [0x62,0x62,0x95,0x10,0x83,0xb2,0x00,0x04,0x00,0x00]
         vpmultishiftqb	xmm30, xmm29, qword ptr [rdx + 1024]{1to2}

# CHECK: vpmultishiftqb	xmm30, xmm29, qword ptr [rdx - 1024]{1to2}
# CHECK: encoding: [0x62,0x62,0x95,0x10,0x83,0x72,0x80]
         vpmultishiftqb	xmm30, xmm29, qword ptr [rdx - 1024]{1to2}

# CHECK: vpmultishiftqb	xmm30, xmm29, qword ptr [rdx - 1032]{1to2}
# CHECK: encoding: [0x62,0x62,0x95,0x10,0x83,0xb2,0xf8,0xfb,0xff,0xff]
         vpmultishiftqb	xmm30, xmm29, qword ptr [rdx - 1032]{1to2}

# CHECK: vpmultishiftqb	ymm30, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x95,0x20,0x83,0xf4]
         vpmultishiftqb	ymm30, ymm29, ymm28

# CHECK: vpmultishiftqb	ymm30 {k7}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x95,0x27,0x83,0xf4]
         vpmultishiftqb	ymm30 {k7}, ymm29, ymm28

# CHECK: vpmultishiftqb	ymm30 {k7} {z}, ymm29, ymm28
# CHECK: encoding: [0x62,0x02,0x95,0xa7,0x83,0xf4]
         vpmultishiftqb	ymm30 {k7} {z}, ymm29, ymm28

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x95,0x20,0x83,0x31]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rcx]

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x95,0x20,0x83,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 291]

# CHECK: vpmultishiftqb	ymm30, ymm29, qword ptr [rcx]{1to4}
# CHECK: encoding: [0x62,0x62,0x95,0x30,0x83,0x31]
         vpmultishiftqb	ymm30, ymm29, qword ptr [rcx]{1to4}

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx + 4064]
# CHECK: encoding: [0x62,0x62,0x95,0x20,0x83,0x72,0x7f]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx + 4064]

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx + 4096]
# CHECK: encoding: [0x62,0x62,0x95,0x20,0x83,0xb2,0x00,0x10,0x00,0x00]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx + 4096]

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx - 4096]
# CHECK: encoding: [0x62,0x62,0x95,0x20,0x83,0x72,0x80]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx - 4096]

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx - 4128]
# CHECK: encoding: [0x62,0x62,0x95,0x20,0x83,0xb2,0xe0,0xef,0xff,0xff]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rdx - 4128]

# CHECK: vpmultishiftqb	ymm30, ymm29, qword ptr [rdx + 1016]{1to4}
# CHECK: encoding: [0x62,0x62,0x95,0x30,0x83,0x72,0x7f]
         vpmultishiftqb	ymm30, ymm29, qword ptr [rdx + 1016]{1to4}

# CHECK: vpmultishiftqb	ymm30, ymm29, qword ptr [rdx + 1024]{1to4}
# CHECK: encoding: [0x62,0x62,0x95,0x30,0x83,0xb2,0x00,0x04,0x00,0x00]
         vpmultishiftqb	ymm30, ymm29, qword ptr [rdx + 1024]{1to4}

# CHECK: vpmultishiftqb	ymm30, ymm29, qword ptr [rdx - 1024]{1to4}
# CHECK: encoding: [0x62,0x62,0x95,0x30,0x83,0x72,0x80]
         vpmultishiftqb	ymm30, ymm29, qword ptr [rdx - 1024]{1to4}

# CHECK: vpmultishiftqb	ymm30, ymm29, qword ptr [rdx - 1032]{1to4}
# CHECK: encoding: [0x62,0x62,0x95,0x30,0x83,0xb2,0xf8,0xfb,0xff,0xff]
         vpmultishiftqb	ymm30, ymm29, qword ptr [rdx - 1032]{1to4}

# CHECK: vpmultishiftqb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x95,0x00,0x83,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpmultishiftqb	xmm30, xmm29, xmmword ptr [rax + 8*r14 + 4660]

# CHECK: vpmultishiftqb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x95,0x20,0x83,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpmultishiftqb	ymm30, ymm29, ymmword ptr [rax + 8*r14 + 4660]

# CHECK: vpmultishiftqb	zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x95,0x40,0x83,0xf4]
         vpmultishiftqb	zmm30, zmm29, zmm28

# CHECK: vpmultishiftqb	zmm30 {k7}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x95,0x47,0x83,0xf4]
         vpmultishiftqb	zmm30 {k7}, zmm29, zmm28

# CHECK: vpmultishiftqb	zmm30 {k7} {z}, zmm29, zmm28
# CHECK: encoding: [0x62,0x02,0x95,0xc7,0x83,0xf4]
         vpmultishiftqb	zmm30 {k7} {z}, zmm29, zmm28

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0x62,0x95,0x40,0x83,0x31]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rcx]

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0x22,0x95,0x40,0x83,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vpmultishiftqb	zmm30, zmm29, qword ptr [rcx]{1to8}
# CHECK: encoding: [0x62,0x62,0x95,0x50,0x83,0x31]
         vpmultishiftqb	zmm30, zmm29, qword ptr [rcx]{1to8}

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx + 8128]
# CHECK: encoding: [0x62,0x62,0x95,0x40,0x83,0x72,0x7f]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx + 8128]

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx + 8192]
# CHECK: encoding: [0x62,0x62,0x95,0x40,0x83,0xb2,0x00,0x20,0x00,0x00]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx + 8192]

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx - 8192]
# CHECK: encoding: [0x62,0x62,0x95,0x40,0x83,0x72,0x80]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx - 8192]

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx - 8256]
# CHECK: encoding: [0x62,0x62,0x95,0x40,0x83,0xb2,0xc0,0xdf,0xff,0xff]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rdx - 8256]

# CHECK: vpmultishiftqb	zmm30, zmm29, qword ptr [rdx + 1016]{1to8}
# CHECK: encoding: [0x62,0x62,0x95,0x50,0x83,0x72,0x7f]
         vpmultishiftqb	zmm30, zmm29, qword ptr [rdx + 1016]{1to8}

# CHECK: vpmultishiftqb	zmm30, zmm29, qword ptr [rdx + 1024]{1to8}
# CHECK: encoding: [0x62,0x62,0x95,0x50,0x83,0xb2,0x00,0x04,0x00,0x00]
         vpmultishiftqb	zmm30, zmm29, qword ptr [rdx + 1024]{1to8}

# CHECK: vpmultishiftqb	zmm30, zmm29, qword ptr [rdx - 1024]{1to8}
# CHECK: encoding: [0x62,0x62,0x95,0x50,0x83,0x72,0x80]
         vpmultishiftqb	zmm30, zmm29, qword ptr [rdx - 1024]{1to8}

# CHECK: vpmultishiftqb	zmm30, zmm29, qword ptr [rdx - 1032]{1to8}
# CHECK: encoding: [0x62,0x62,0x95,0x50,0x83,0xb2,0xf8,0xfb,0xff,0xff]
         vpmultishiftqb	zmm30, zmm29, qword ptr [rdx - 1032]{1to8}

# CHECK: vpmultishiftqb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]
# CHECK: encoding: [0x62,0x22,0x95,0x40,0x83,0xb4,0xf0,0x34,0x12,0x00,0x00]
         vpmultishiftqb	zmm30, zmm29, zmmword ptr [rax + 8*r14 + 4660]
