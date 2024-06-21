# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

## AddRegFrm

# CHECK: mov	r16d, 1
# CHECK: encoding: [0xd5,0x10,0xb8,0x01,0x00,0x00,0x00]
         mov	r16d, 1

## MRMSrcReg

# CHECK: movsxd	rax, r16d
# CHECK: encoding: [0xd5,0x18,0x63,0xc0]
         movsxd	rax, r16d

# CHECK: movsxd	r16, eax
# CHECK: encoding: [0xd5,0x48,0x63,0xc0]
         movsxd	r16, eax

# CHECK: movsxd	r17, r16d
# CHECK: encoding: [0xd5,0x58,0x63,0xc8]
         movsxd	r17, r16d

# CHECK: popcnt r17d, r16d
# CHECK: encoding: [0xf3,0xd5,0xd0,0xb8,0xc8]
         popcnt r17d, r16d

## MRMSrcRegCC

# CHECK: cmovl	eax, r16d
# CHECK: encoding: [0xd5,0x90,0x4c,0xc0]
         cmovl	eax, r16d

# CHECK: cmovl	r16d, eax
# CHECK: encoding: [0xd5,0xc0,0x4c,0xc0]
         cmovl	r16d, eax

# CHECK: cmovl	r17d, r16d
# CHECK: encoding: [0xd5,0xd0,0x4c,0xc8]
         cmovl	r17d, r16d

## MRMSrcMem

# CHECK: imul	ebx, dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x90,0xaf,0x1c,0x00]
         imul	ebx, dword ptr [r16 + rax]

# CHECK: imul	ebx, dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0xa0,0xaf,0x1c,0x00]
         imul	ebx, dword ptr [rax + r16]

# CHECK: imul	r16d, dword ptr [rax + rbx]
# CHECK: encoding: [0xd5,0xc0,0xaf,0x04,0x18]
         imul	r16d, dword ptr [rax + rbx]

# CHECK: imul	eax, dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xb0,0xaf,0x04,0x08]
         imul	eax, dword ptr [r16 + r17]

# CHECK: imul	r17d, dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0xe0,0xaf,0x0c,0x00]
         imul	r17d, dword ptr [rax + r16]

# CHECK: imul	r17d, dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0xd0,0xaf,0x0c,0x00]
         imul	r17d, dword ptr [r16 + rax]

# CHECK: imul	r18d, dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xf0,0xaf,0x14,0x08]
         imul	r18d, dword ptr [r16 + r17]

## MRMSrcMemCC

# CHECK: cmovl	ebx, dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x90,0x4c,0x1c,0x00]
         cmovl	ebx, dword ptr [r16 + rax]

# CHECK: cmovl	ebx, dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0xa0,0x4c,0x1c,0x00]
         cmovl	ebx, dword ptr [rax + r16]

# CHECK: cmovl	r16d, dword ptr [rax + rbx]
# CHECK: encoding: [0xd5,0xc0,0x4c,0x04,0x18]
         cmovl	r16d, dword ptr [rax + rbx]

# CHECK: cmovl	eax, dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xb0,0x4c,0x04,0x08]
         cmovl	eax, dword ptr [r16 + r17]

# CHECK: cmovl	r17d, dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0xe0,0x4c,0x0c,0x00]
         cmovl	r17d, dword ptr [rax + r16]

# CHECK: cmovl	r17d, dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0xd0,0x4c,0x0c,0x00]
         cmovl	r17d, dword ptr [r16 + rax]

# CHECK: cmovl	r18d, dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xf0,0x4c,0x14,0x08]
         cmovl	r18d, dword ptr [r16 + r17]

## MRMDestReg

# CHECK: mov	r16d, eax
# CHECK: encoding: [0xd5,0x10,0x89,0xc0]
         mov	r16d, eax

# CHECK: mov	eax, r16d
# CHECK: encoding: [0xd5,0x40,0x89,0xc0]
         mov	eax, r16d

# CHECK: mov	r17d, r16d
# CHECK: encoding: [0xd5,0x50,0x89,0xc1]
         mov	r17d, r16d

## MRMDestMem

# CHECK: mov	dword ptr [r16 + rax], ebx
# CHECK: encoding: [0xd5,0x10,0x89,0x1c,0x00]
         mov	dword ptr [r16 + rax], ebx

# CHECK: mov	dword ptr [rax + r16], ebx
# CHECK: encoding: [0xd5,0x20,0x89,0x1c,0x00]
         mov	dword ptr [rax + r16], ebx

# CHECK: mov	dword ptr [rax + rbx], r16d
# CHECK: encoding: [0xd5,0x40,0x89,0x04,0x18]
         mov	dword ptr [rax + rbx], r16d

# CHECK: mov	dword ptr [r16 + r17], eax
# CHECK: encoding: [0xd5,0x30,0x89,0x04,0x08]
         mov	dword ptr [r16 + r17], eax

# CHECK: mov	dword ptr [rax + r16], r17d
# CHECK: encoding: [0xd5,0x60,0x89,0x0c,0x00]
         mov	dword ptr [rax + r16], r17d

# CHECK: mov	dword ptr [r16 + rax], r17d
# CHECK: encoding: [0xd5,0x50,0x89,0x0c,0x00]
         mov	dword ptr [r16 + rax], r17d

# CHECK: mov	dword ptr [r16 + r17], r18d
# CHECK: encoding: [0xd5,0x70,0x89,0x14,0x08]
         mov	dword ptr [r16 + r17], r18d

# CHECK: mov    byte ptr [r16 + r14], bpl
# CHECK: encoding: [0xd5,0x12,0x88,0x2c,0x30]
         mov    byte ptr [r16 + r14], bpl

## MRMXmCC

# CHECK: sete	byte ptr [rax + r16]
# CHECK: encoding: [0xd5,0xa0,0x94,0x04,0x00]
         sete	byte ptr [rax + r16]

# CHECK: sete	byte ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x90,0x94,0x04,0x00]
         sete	byte ptr [r16 + rax]

# CHECK: sete	byte ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xb0,0x94,0x04,0x08]
         sete	byte ptr [r16 + r17]

## MRMXm

# CHECK: nop	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0xa0,0x1f,0x04,0x00]
         nop	dword ptr [rax + r16]

# CHECK: nop	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x90,0x1f,0x04,0x00]
         nop	dword ptr [r16 + rax]

# CHECK: nop	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0xb0,0x1f,0x04,0x08]
         nop	dword ptr [r16 + r17]

## MRM0m

# CHECK: inc	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xff,0x04,0x00]
         inc	dword ptr [rax + r16]

# CHECK: inc	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xff,0x04,0x00]
         inc	dword ptr [r16 + rax]

# CHECK: inc	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xff,0x04,0x08]
         inc	dword ptr [r16 + r17]

## MRM1m

# CHECK: dec	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xff,0x0c,0x00]
         dec	dword ptr [rax + r16]

# CHECK: dec	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xff,0x0c,0x00]
         dec	dword ptr [r16 + rax]

# CHECK: dec	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xff,0x0c,0x08]
         dec	dword ptr [r16 + r17]

## MRM2m

# CHECK: not	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x14,0x00]
         not	dword ptr [rax + r16]

# CHECK: not	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x14,0x00]
         not	dword ptr [r16 + rax]

# CHECK: not	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x14,0x08]
         not	dword ptr [r16 + r17]

## MRM3m

# CHECK: neg	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x1c,0x00]
         neg	dword ptr [rax + r16]

# CHECK: neg	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x1c,0x00]
         neg	dword ptr [r16 + rax]

# CHECK: neg	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x1c,0x08]
         neg	dword ptr [r16 + r17]

## MRM4m

# CHECK: mul	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x24,0x00]
         mul	dword ptr [rax + r16]

# CHECK: mul	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x24,0x00]
         mul	dword ptr [r16 + rax]

# CHECK: mul	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x24,0x08]
         mul	dword ptr [r16 + r17]

## MRM5m

# CHECK: imul	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x2c,0x00]
         imul	dword ptr [rax + r16]

# CHECK: imul	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x2c,0x00]
         imul	dword ptr [r16 + rax]

# CHECK: imul	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x2c,0x08]
         imul	dword ptr [r16 + r17]

## MRM6m

# CHECK: div	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x34,0x00]
         div	dword ptr [rax + r16]

# CHECK: div	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x34,0x00]
         div	dword ptr [r16 + rax]

# CHECK: div	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x34,0x08]
         div	dword ptr [r16 + r17]

## MRM7m

# CHECK: idiv	dword ptr [rax + r16]
# CHECK: encoding: [0xd5,0x20,0xf7,0x3c,0x00]
         idiv	dword ptr [rax + r16]

# CHECK: idiv	dword ptr [r16 + rax]
# CHECK: encoding: [0xd5,0x10,0xf7,0x3c,0x00]
         idiv	dword ptr [r16 + rax]

# CHECK: idiv	dword ptr [r16 + r17]
# CHECK: encoding: [0xd5,0x30,0xf7,0x3c,0x08]
         idiv	dword ptr [r16 + r17]

## MRMXrCC

# CHECK: sete	r16b
# CHECK: encoding: [0xd5,0x90,0x94,0xc0]
         sete	r16b

## MRMXr

# CHECK: nop	r16d
# CHECK: encoding: [0xd5,0x90,0x1f,0xc0]
         nop	r16d

## MRM0r

# CHECK: inc	r16d
# CHECK: encoding: [0xd5,0x10,0xff,0xc0]
         inc	r16d

## MRM1r

# CHECK: dec	r16d
# CHECK: encoding: [0xd5,0x10,0xff,0xc8]
         dec	r16d

## MRM2r

# CHECK: not	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xd0]
         not	r16d

## MRM3r

# CHECK: neg	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xd8]
         neg	r16d

## MRM4r

# CHECK: mul	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xe0]
         mul	r16d

## MRM5r

# CHECK: imul	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xe8]
         imul	r16d

## MRM6r

# CHECK: div	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xf0]
         div	r16d

## MRM7r

# CHECK: idiv	r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xf8]
         idiv	r16d
