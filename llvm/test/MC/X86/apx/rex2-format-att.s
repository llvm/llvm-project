# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

## AddRegFrm

# CHECK: movl	$1, %r16d
# CHECK: encoding: [0xd5,0x10,0xb8,0x01,0x00,0x00,0x00]
         movl	$1, %r16d

## MRMSrcReg

# CHECK: movslq	%r16d, %rax
# CHECK: encoding: [0xd5,0x18,0x63,0xc0]
         movslq	%r16d, %rax

# CHECK: movslq	%eax, %r16
# CHECK: encoding: [0xd5,0x48,0x63,0xc0]
         movslq	%eax, %r16

# CHECK: movslq	%r16d, %r17
# CHECK: encoding: [0xd5,0x58,0x63,0xc8]
         movslq	%r16d, %r17

## MRMSrcRegCC

# CHECK: cmovll	%r16d, %eax
# CHECK: encoding: [0xd5,0x90,0x4c,0xc0]
         cmovll	%r16d, %eax

# CHECK: cmovll	%eax, %r16d
# CHECK: encoding: [0xd5,0xc0,0x4c,0xc0]
         cmovll	%eax, %r16d

# CHECK: cmovll	%r16d, %r17d
# CHECK: encoding: [0xd5,0xd0,0x4c,0xc8]
         cmovll	%r16d, %r17d

## MRMSrcMem

# CHECK: imull	(%r16,%rax), %ebx
# CHECK: encoding: [0xd5,0x90,0xaf,0x1c,0x00]
         imull	(%r16,%rax), %ebx

# CHECK: imull	(%rax,%r16), %ebx
# CHECK: encoding: [0xd5,0xa0,0xaf,0x1c,0x00]
         imull	(%rax,%r16), %ebx

# CHECK: imull	(%rax,%rbx), %r16d
# CHECK: encoding: [0xd5,0xc0,0xaf,0x04,0x18]
         imull	(%rax,%rbx), %r16d

# CHECK: imull	(%r16,%r17), %eax
# CHECK: encoding: [0xd5,0xb0,0xaf,0x04,0x08]
         imull	(%r16,%r17), %eax

# CHECK: imull	(%rax,%r16), %r17d
# CHECK: encoding: [0xd5,0xe0,0xaf,0x0c,0x00]
         imull	(%rax,%r16), %r17d

# CHECK: imull	(%r16,%rax), %r17d
# CHECK: encoding: [0xd5,0xd0,0xaf,0x0c,0x00]
         imull	(%r16,%rax), %r17d

# CHECK: imull	(%r16,%r17), %r18d
# CHECK: encoding: [0xd5,0xf0,0xaf,0x14,0x08]
         imull	(%r16,%r17), %r18d

## MRMSrcMemCC

# CHECK: cmovll	(%r16,%rax), %ebx
# CHECK: encoding: [0xd5,0x90,0x4c,0x1c,0x00]
         cmovll	(%r16,%rax), %ebx

# CHECK: cmovll	(%rax,%r16), %ebx
# CHECK: encoding: [0xd5,0xa0,0x4c,0x1c,0x00]
         cmovll	(%rax,%r16), %ebx

# CHECK: cmovll	(%rax,%rbx), %r16d
# CHECK: encoding: [0xd5,0xc0,0x4c,0x04,0x18]
         cmovll	(%rax,%rbx), %r16d

# CHECK: cmovll	(%r16,%r17), %eax
# CHECK: encoding: [0xd5,0xb0,0x4c,0x04,0x08]
         cmovll	(%r16,%r17), %eax

# CHECK: cmovll	(%rax,%r16), %r17d
# CHECK: encoding: [0xd5,0xe0,0x4c,0x0c,0x00]
         cmovll	(%rax,%r16), %r17d

# CHECK: cmovll	(%r16,%rax), %r17d
# CHECK: encoding: [0xd5,0xd0,0x4c,0x0c,0x00]
         cmovll	(%r16,%rax), %r17d

# CHECK: cmovll	(%r16,%r17), %r18d
# CHECK: encoding: [0xd5,0xf0,0x4c,0x14,0x08]
         cmovll	(%r16,%r17), %r18d

## MRMDestReg

# CHECK: movl	%eax, %r16d
# CHECK: encoding: [0xd5,0x10,0x89,0xc0]
         movl	%eax, %r16d

# CHECK: movl	%r16d, %eax
# CHECK: encoding: [0xd5,0x40,0x89,0xc0]
         movl	%r16d, %eax

# CHECK: movl	%r16d, %r17d
# CHECK: encoding: [0xd5,0x50,0x89,0xc1]
         movl	%r16d, %r17d

## MRMDestMem

# CHECK: movl	%ebx, (%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0x89,0x1c,0x00]
         movl	%ebx, (%r16,%rax)

# CHECK: movl	%ebx, (%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0x89,0x1c,0x00]
         movl	%ebx, (%rax,%r16)

# CHECK: movl	%r16d, (%rax,%rbx)
# CHECK: encoding: [0xd5,0x40,0x89,0x04,0x18]
         movl	%r16d, (%rax,%rbx)

# CHECK: movl	%eax, (%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0x89,0x04,0x08]
         movl	%eax, (%r16,%r17)

# CHECK: movl	%r17d, (%rax,%r16)
# CHECK: encoding: [0xd5,0x60,0x89,0x0c,0x00]
         movl	%r17d, (%rax,%r16)

# CHECK: movl	%r17d, (%r16,%rax)
# CHECK: encoding: [0xd5,0x50,0x89,0x0c,0x00]
         movl	%r17d, (%r16,%rax)

# CHECK: movl	%r18d, (%r16,%r17)
# CHECK: encoding: [0xd5,0x70,0x89,0x14,0x08]
         movl	%r18d, (%r16,%r17)

# CHECK: movb    %bpl, (%r16,%r14)
# CHECK: encoding: [0xd5,0x12,0x88,0x2c,0x30]
         movb    %bpl, (%r16,%r14)

## MRMXmCC

# CHECK: sete	(%rax,%r16)
# CHECK: encoding: [0xd5,0xa0,0x94,0x04,0x00]
         sete	(%rax,%r16)

# CHECK: sete	(%r16,%rax)
# CHECK: encoding: [0xd5,0x90,0x94,0x04,0x00]
         sete	(%r16,%rax)

# CHECK: sete	(%r16,%r17)
# CHECK: encoding: [0xd5,0xb0,0x94,0x04,0x08]
         sete	(%r16,%r17)

## MRMXm

# CHECK: nopl	(%rax,%r16)
# CHECK: encoding: [0xd5,0xa0,0x1f,0x04,0x00]
         nopl	(%rax,%r16)

# CHECK: nopl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x90,0x1f,0x04,0x00]
         nopl	(%r16,%rax)

# CHECK: nopl	(%r16,%r17)
# CHECK: encoding: [0xd5,0xb0,0x1f,0x04,0x08]
         nopl	(%r16,%r17)

## MRM0m

# CHECK: incl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xff,0x04,0x00]
         incl	(%rax,%r16)

# CHECK: incl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xff,0x04,0x00]
         incl	(%r16,%rax)

# CHECK: incl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xff,0x04,0x08]
         incl	(%r16,%r17)

## MRM1m

# CHECK: decl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xff,0x0c,0x00]
         decl	(%rax,%r16)

# CHECK: decl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xff,0x0c,0x00]
         decl	(%r16,%rax)

# CHECK: decl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xff,0x0c,0x08]
         decl	(%r16,%r17)

## MRM2m

# CHECK: notl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x14,0x00]
         notl	(%rax,%r16)

# CHECK: notl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x14,0x00]
         notl	(%r16,%rax)

# CHECK: notl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x14,0x08]
         notl	(%r16,%r17)

## MRM3m

# CHECK: negl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x1c,0x00]
         negl	(%rax,%r16)

# CHECK: negl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x1c,0x00]
         negl	(%r16,%rax)

# CHECK: negl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x1c,0x08]
         negl	(%r16,%r17)

## MRM4m

# CHECK: mull	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x24,0x00]
         mull	(%rax,%r16)

# CHECK: mull	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x24,0x00]
         mull	(%r16,%rax)

# CHECK: mull	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x24,0x08]
         mull	(%r16,%r17)

## MRM5m

# CHECK: imull	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x2c,0x00]
         imull	(%rax,%r16)

# CHECK: imull	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x2c,0x00]
         imull	(%r16,%rax)

# CHECK: imull	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x2c,0x08]
         imull	(%r16,%r17)

## MRM6m

# CHECK: divl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x34,0x00]
         divl	(%rax,%r16)

# CHECK: divl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x34,0x00]
         divl	(%r16,%rax)

# CHECK: divl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x34,0x08]
         divl	(%r16,%r17)

## MRM7m

# CHECK: idivl	(%rax,%r16)
# CHECK: encoding: [0xd5,0x20,0xf7,0x3c,0x00]
         idivl	(%rax,%r16)

# CHECK: idivl	(%r16,%rax)
# CHECK: encoding: [0xd5,0x10,0xf7,0x3c,0x00]
         idivl	(%r16,%rax)

# CHECK: idivl	(%r16,%r17)
# CHECK: encoding: [0xd5,0x30,0xf7,0x3c,0x08]
         idivl	(%r16,%r17)

## MRMXrCC

# CHECK: sete	%r16b
# CHECK: encoding: [0xd5,0x90,0x94,0xc0]
         sete	%r16b

## MRMXr

# CHECK: nopl	%r16d
# CHECK: encoding: [0xd5,0x90,0x1f,0xc0]
         nopl	%r16d

## MRM0r

# CHECK: incl	%r16d
# CHECK: encoding: [0xd5,0x10,0xff,0xc0]
         incl	%r16d

## MRM1r

# CHECK: decl	%r16d
# CHECK: encoding: [0xd5,0x10,0xff,0xc8]
         decl	%r16d

## MRM2r

# CHECK: notl	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xd0]
         notl	%r16d

## MRM3r

# CHECK: negl	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xd8]
         negl	%r16d

## MRM4r

# CHECK: mull	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xe0]
         mull	%r16d

## MRM5r

# CHECK: imull	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xe8]
         imull	%r16d

## MRM6r

# CHECK: divl	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xf0]
         divl	%r16d

## MRM7r

# CHECK: idivl	%r16d
# CHECK: encoding: [0xd5,0x10,0xf7,0xf8]
         idivl	%r16d
