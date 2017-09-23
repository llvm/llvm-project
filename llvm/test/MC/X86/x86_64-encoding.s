// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// PR7195
// CHECK: callw 42
// CHECK: encoding: [0x66,0xe8,A,A]
       callw 42

// rdar://8127102
// CHECK: movq	%gs:(%rdi), %rax
// CHECK: encoding: [0x65,0x48,0x8b,0x07]
movq	%gs:(%rdi), %rax

// CHECK: crc32b 	%bl, %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0xc3]
        crc32b	%bl, %eax

// CHECK: crc32b 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %eax

// CHECK: crc32w 	%bx, %eax
// CHECK:  encoding: [0x66,0xf2,0x0f,0x38,0xf1,0xc3]
        crc32w	%bx, %eax

// CHECK: crc32w 	4(%rbx), %eax
// CHECK:  encoding: [0x66,0xf2,0x0f,0x38,0xf1,0x43,0x04]
        crc32w	4(%rbx), %eax

// CHECK: crc32l 	%ebx, %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0xc3]
        crc32l	%ebx, %eax

// CHECK: crc32l 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x43,0x04]
        crc32l	4(%rbx), %eax

// CHECK: crc32l 	3735928559(%rbx,%rcx,8), %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x8c,0xcb,0xef,0xbe,0xad,0xde]
        	crc32l   0xdeadbeef(%rbx,%rcx,8),%ecx

// CHECK: crc32l 	69, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0x45,0x00,0x00,0x00]
        	crc32l   0x45,%ecx

// CHECK: crc32l 	32493, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0xed,0x7e,0x00,0x00]
        	crc32l   0x7eed,%ecx

// CHECK: crc32l 	3133065982, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0x0c,0x25,0xfe,0xca,0xbe,0xba]
        	crc32l   0xbabecafe,%ecx

// CHECK: crc32l 	%ecx, %ecx
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf1,0xc9]
        	crc32l   %ecx,%ecx

// CHECK: crc32b 	%r11b, %eax
// CHECK:  encoding: [0xf2,0x41,0x0f,0x38,0xf0,0xc3]
        crc32b	%r11b, %eax

// CHECK: crc32b 	4(%rbx), %eax
// CHECK:  encoding: [0xf2,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %eax

// CHECK: crc32b 	%dil, %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf0,0xc7]
        crc32b	%dil,%rax

// CHECK: crc32b 	%r11b, %rax
// CHECK:  encoding: [0xf2,0x49,0x0f,0x38,0xf0,0xc3]
        crc32b	%r11b,%rax

// CHECK: crc32b 	4(%rbx), %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf0,0x43,0x04]
        crc32b	4(%rbx), %rax

// CHECK: crc32q 	%rbx, %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf1,0xc3]
        crc32q	%rbx, %rax

// CHECK: crc32q 	4(%rbx), %rax
// CHECK:  encoding: [0xf2,0x48,0x0f,0x38,0xf1,0x43,0x04]
        crc32q	4(%rbx), %rax

// CHECK: movd %r8, %mm1
// CHECK:  encoding: [0x49,0x0f,0x6e,0xc8]
movd %r8, %mm1

// CHECK: movd %r8d, %mm1
// CHECK:  encoding: [0x41,0x0f,0x6e,0xc8]
movd %r8d, %mm1

// CHECK: movd %rdx, %mm1
// CHECK:  encoding: [0x48,0x0f,0x6e,0xca]
movd %rdx, %mm1

// CHECK: movd %edx, %mm1
// CHECK:  encoding: [0x0f,0x6e,0xca]
movd %edx, %mm1

// CHECK: movd %mm1, %r8
// CHECK:  encoding: [0x49,0x0f,0x7e,0xc8]
movd %mm1, %r8

// CHECK: movd %mm1, %r8d
// CHECK:  encoding: [0x41,0x0f,0x7e,0xc8]
movd %mm1, %r8d

// CHECK: movd %mm1, %rdx
// CHECK:  encoding: [0x48,0x0f,0x7e,0xca]
movd %mm1, %rdx

// CHECK: movd %mm1, %edx
// CHECK:  encoding: [0x0f,0x7e,0xca]
movd %mm1, %edx

// rdar://7840289
// CHECK: pshufb	CPI1_0(%rip), %xmm1
// CHECK:  encoding: [0x66,0x0f,0x38,0x00,0x0d,A,A,A,A]
// CHECK:  fixup A - offset: 5, value: CPI1_0-4
pshufb	CPI1_0(%rip), %xmm1

// CHECK: sha1rnds4 $1, %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x3a,0xcc,0xd1,0x01]
sha1rnds4 $1, %xmm1, %xmm2

// CHECK: sha1rnds4 $1, (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x3a,0xcc,0x10,0x01]
sha1rnds4 $1, (%rax), %xmm2

// CHECK: sha1nexte %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xc8,0xd1]
sha1nexte %xmm1, %xmm2

// CHECK: sha1msg1 %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xc9,0xd1]
sha1msg1 %xmm1, %xmm2

// CHECK: sha1msg1 (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xc9,0x10]
sha1msg1 (%rax), %xmm2

// CHECK: sha1msg2 %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xca,0xd1]
sha1msg2 %xmm1, %xmm2

// CHECK: sha1msg2 (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xca,0x10]
sha1msg2 (%rax), %xmm2

// CHECK: sha256rnds2 %xmm0, (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcb,0x10]
sha256rnds2 (%rax), %xmm2

// CHECK: sha256rnds2 %xmm0, %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcb,0xd1]
sha256rnds2 %xmm1, %xmm2

// CHECK: sha256rnds2 %xmm0, (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcb,0x10]
sha256rnds2 %xmm0, (%rax), %xmm2

// CHECK: sha256rnds2 %xmm0, %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcb,0xd1]
sha256rnds2 %xmm0, %xmm1, %xmm2

// CHECK: sha256msg1 %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcc,0xd1]
sha256msg1 %xmm1, %xmm2

// CHECK: sha256msg1 (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcc,0x10]
sha256msg1 (%rax), %xmm2

// CHECK: sha256msg2 %xmm1, %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcd,0xd1]
sha256msg2 %xmm1, %xmm2

// CHECK: sha256msg2 (%rax), %xmm2
// CHECK:   encoding: [0x0f,0x38,0xcd,0x10]
sha256msg2 (%rax), %xmm2

// CHECK: movq  57005(,%riz), %rbx
// CHECK: encoding: [0x48,0x8b,0x1c,0x25,0xad,0xde,0x00,0x00]
          movq  57005(,%riz), %rbx

// CHECK: movq  48879(,%riz), %rax
// CHECK: encoding: [0x48,0x8b,0x04,0x25,0xef,0xbe,0x00,0x00]
          movq  48879(,%riz), %rax

// CHECK: movq  -4(,%riz,8), %rax
// CHECK: encoding: [0x48,0x8b,0x04,0xe5,0xfc,0xff,0xff,0xff]
          movq  -4(,%riz,8), %rax

// CHECK: movq  (%rcx,%riz), %rax
// CHECK: encoding: [0x48,0x8b,0x04,0x21]
          movq  (%rcx,%riz), %rax

// CHECK: movq  (%rcx,%riz,8), %rax
// CHECK: encoding: [0x48,0x8b,0x04,0xe1]
          movq  (%rcx,%riz,8), %rax

// CHECK: fxsave64 (%rax)
// CHECK: encoding: [0x48,0x0f,0xae,0x00]
          fxsaveq (%rax)

// CHECK: fxsave64 (%rax)
// CHECK: encoding: [0x48,0x0f,0xae,0x00]
          fxsave64 (%rax)

// CHECK: fxrstor64 (%rax)
// CHECK: encoding: [0x48,0x0f,0xae,0x08]
          fxrstorq (%rax)

// CHECK: fxrstor64 (%rax)
// CHECK: encoding: [0x48,0x0f,0xae,0x08]
          fxrstor64 (%rax)

// CHECK: leave
// CHECK:  encoding: [0xc9]
        	leave

// CHECK: leave
// CHECK:  encoding: [0xc9]
        	leaveq

// CHECK: flds	(%edi)
// CHECK:  encoding: [0x67,0xd9,0x07]
        	flds	(%edi)

// CHECK: filds	(%edi)
// CHECK:  encoding: [0x67,0xdf,0x07]
        	filds	(%edi)

// CHECK: flds	(%rdi)
// CHECK:  encoding: [0xd9,0x07]
        	flds	(%rdi)

// CHECK: filds	(%rdi)
// CHECK:  encoding: [0xdf,0x07]
        	filds	(%rdi)

// CHECK: pmovmskb	%xmm5, %ecx
// CHECK:  encoding: [0x66,0x0f,0xd7,0xcd]
        	pmovmskb	%xmm5,%rcx

// CHECK: pinsrw $3, %ecx, %xmm5
// CHECK: encoding: [0x66,0x0f,0xc4,0xe9,0x03]
          pinsrw $3, %ecx, %xmm5

// CHECK: pinsrw $3, %ecx, %xmm5
// CHECK: encoding: [0x66,0x0f,0xc4,0xe9,0x03]
          pinsrw $3, %rcx, %xmm5

//CHECK   movq	12(%rdi), %rsi
//CHECK   encoding: [0x48,0x8b,0x77,0x0c]
    movq 	16+0-4(%rdi),%rsi

//CHECK   movq	12(%rdi), %rsi
//CHECK   encoding: [0x48,0x8b,0x77,0x0c]
    movq 	(16+(0-4))(%rdi),%rsi

//CHECK   movq	12(%rdi), %rsi
//CHECK   encoding: [0x48,0x8b,0x77,0x0c]
    movq 	(16+0)-1+1-2+2-3+3-4+4-5+5-6+6-(4)(%rdi),%rsi
