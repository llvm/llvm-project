// RUN: llvm-mc -triple x86_64-unknown-unknown -show-encoding %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

	monitor
// CHECK: monitor
// CHECK: encoding: [0x0f,0x01,0xc8]
	monitor %rax, %rcx, %rdx
// CHECK: monitor
// CHECK: encoding: [0x0f,0x01,0xc8]
	mwait
// CHECK: mwait
// CHECK: encoding: [0x0f,0x01,0xc9]
	mwait %rax, %rcx
// CHECK: mwait
// CHECK: encoding: [0x0f,0x01,0xc9]

// Suffix inference:

// CHECK: addl $0, %eax
        add $0, %eax
// CHECK: addb $255, %al
        add $0xFF, %al
// CHECK: orq %rax, %rdx
        or %rax, %rdx
// CHECK: shlq $3, %rax
        shl $3, %rax


// CHECK: subb %al, %al
        subb %al, %al

// CHECK: addl $24, %eax
        addl $24, %eax

// CHECK: movl %eax, 10(%ebp)
        movl %eax, 10(%ebp)
// CHECK: movl %eax, 10(%ebp,%ebx)
        movl %eax, 10(%ebp, %ebx)
// CHECK: movl %eax, 10(%ebp,%ebx,4)
        movl %eax, 10(%ebp, %ebx, 4)
// CHECK: movl %eax, 10(,%ebx,4)
        movl %eax, 10(, %ebx, 4)

// CHECK: movl 0, %eax        
        movl 0, %eax
// CHECK: movl $0, %eax        
        movl $0, %eax
        
// CHECK: ret
        ret
        
// CHECK: retw
        retw
        
// FIXME: Check that this matches SUB32ri8
// CHECK: subl $1, %eax
        subl $1, %eax
        
// FIXME: Check that this matches SUB32ri8
// CHECK: subl $-1, %eax
        subl $-1, %eax
        
// FIXME: Check that this matches SUB32ri
// CHECK: subl $256, %eax
        subl $256, %eax

// FIXME: Check that this matches XOR64ri8
// CHECK: xorq $1, %rax
        xorq $1, %rax
        
// FIXME: Check that this matches XOR64ri32
// CHECK: xorq $256, %rax
        xorq $256, %rax

// FIXME: Check that this matches SUB8rr
// CHECK: subb %al, %bl
        subb %al, %bl

// FIXME: Check that this matches SUB16rr
// CHECK: subw %ax, %bx
        subw %ax, %bx
        
// FIXME: Check that this matches SUB32rr
// CHECK: subl %eax, %ebx
        subl %eax, %ebx
        
// FIXME: Check that this matches the correct instruction.
// CHECK: callq *%rax
        call *%rax

// FIXME: Check that this matches the correct instruction.
// CHECK: shldl %cl, %eax, %ebx
        shldl %cl, %eax, %ebx

// CHECK: shll $2, %eax
        shll $2, %eax

// CHECK: shll $2, %eax
        sall $2, %eax

// CHECK: rep
// CHECK-NEXT: movsb
rep     # comment
movsb

// CHECK: rep
// CHECK: insb
        rep;insb

// CHECK: rep
// CHECK: outsb
        rep;outsb

// CHECK: rep
// CHECK: movsb
        rep;movsb


// rdar://8470918
smovb // CHECK: movsb
smovw // CHECK: movsw
smovl // CHECK: movsl
smovq // CHECK: movsq

// rdar://8456361
// CHECK: rep
// CHECK: movsl
        rep movsd

// CHECK: rep
// CHECK: lodsb
        rep;lodsb

// CHECK: rep
// CHECK: stosb
        rep;stosb

// NOTE: repz and repe have the same opcode as rep
// CHECK: rep
// CHECK: cmpsb
        repz;cmpsb

// NOTE: repnz has the same opcode as repne
// CHECK: repne
// CHECK: cmpsb
        repnz;cmpsb

// NOTE: repe and repz have the same opcode as rep
// CHECK: rep
// CHECK: scasb
        repe;scasb

// CHECK: repne
// CHECK: scasb
        repne;scasb

// CHECK: lock
// CHECK: cmpxchgb %al, (%ebx)
        lock;cmpxchgb %al, 0(%ebx)

// CHECK: cs
// CHECK: movb (%eax), %al
        cs;movb 0(%eax), %al

// CHECK: ss
// CHECK: movb (%eax), %al
        ss;movb 0(%eax), %al

// CHECK: ds
// CHECK: movb (%eax), %al
        ds;movb 0(%eax), %al

// CHECK: es
// CHECK: movb (%eax), %al
        es;movb 0(%eax), %al

// CHECK: fs
// CHECK: movb (%eax), %al
        fs;movb 0(%eax), %al

// CHECK: gs
// CHECK: movb (%eax), %al
        gs;movb 0(%eax), %al

// CHECK: fadd %st(0)
// CHECK: fadd %st(1)
// CHECK: fadd %st(7)

fadd %st(0)
fadd %st(1)
fadd %st(7)

// CHECK: leal 0, %eax
        leal 0, %eax

// rdar://7986634 - Insensitivity on opcodes.
// CHECK: int3
INT3

// rdar://8735979 - int $3 -> int3
// CHECK: int3
int	$3


// Allow scale factor without index register.
// CHECK: movaps	%xmm3, (%esi)
// CHECK-STDERR: warning: scale factor without index register is ignored
movaps %xmm3, (%esi, 2)

// CHECK: imull $12, %eax
imul $12, %eax

// CHECK: imull %ecx, %eax
imull %ecx, %eax


// rdar://8208481
// CHECK: outb	%al, $161
outb	%al, $161
// CHECK: outw	%ax, $128
outw	%ax, $128
// CHECK: inb	$161, %al
inb	$161, %al

// rdar://8017621
// CHECK: pushq	$1
push $1

// rdar://9716860
pushq $1
// CHECK: encoding: [0x6a,0x01]
pushq $1111111
// CHECK: encoding: [0x68,0x47,0xf4,0x10,0x00]

// rdar://8017530
// CHECK: sldtw	4
sldt	4

// rdar://8208499
// CHECK: cmovnew	%bx, %ax
cmovnz %bx, %ax
// CHECK: cmovneq	%rbx, %rax
cmovnzq %rbx, %rax


// rdar://8407928
// CHECK: inb	$127, %al
// CHECK: inw	%dx, %ax
// CHECK: outb	%al, $127
// CHECK: outw	%ax, %dx
// CHECK: inl	%dx, %eax
inb	$0x7f
inw	%dx
outb	$0x7f
outw	%dx
inl	%dx


// PR8114
// CHECK: outb	%al, %dx
// CHECK: outb	%al, %dx
// CHECK: outw	%ax, %dx
// CHECK: outw	%ax, %dx
// CHECK: outl	%eax, %dx
// CHECK: outl	%eax, %dx

out	%al, (%dx)
outb	%al, (%dx)
out	%ax, (%dx)
outw	%ax, (%dx)
out	%eax, (%dx)
outl	%eax, (%dx)

// CHECK: inb	%dx, %al
// CHECK: inb	%dx, %al
// CHECK: inw	%dx, %ax
// CHECK: inw	%dx, %ax
// CHECK: inl	%dx, %eax
// CHECK: inl	%dx, %eax

in	(%dx), %al
inb	(%dx), %al
in	(%dx), %ax
inw	(%dx), %ax
in	(%dx), %eax
inl	(%dx), %eax

//PR15455

outsb	(%rsi), (%dx)
// CHECK: outsb	(%rsi), %dx
outsw	(%rsi), (%dx)
// CHECK: outsw	(%rsi), %dx
outsl	(%rsi), (%dx)
// CHECK: outsl	(%rsi), %dx

insb	(%dx), %es:(%rdi)
// CHECK: insb	%dx, %es:(%rdi)
insw	(%dx), %es:(%rdi)
// CHECK: insw	%dx, %es:(%rdi)
insl	(%dx), %es:(%rdi)
// CHECK: insl	%dx, %es:(%rdi)	

// rdar://8431422

// CHECK: fxch %st(1)
// CHECK: fucom %st(1)
// CHECK: fucomp %st(1)
// CHECK: faddp %st, %st(1)
// CHECK: faddp %st, %st(0)
// CHECK: fsubp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fdivp %st, %st(1)
// CHECK: fdivrp %st, %st(1)

fxch
fucom
fucomp
faddp
faddp %st
fsubp
fsubrp
fmulp
fdivp
fdivrp

// CHECK: fcomi %st(1)
// CHECK: fcomi	%st(2)
// CHECK: fucomi %st(1)
// CHECK: fucomi %st(2)
// CHECK: fucomi %st(2)

fcomi
fcomi	%st(2)
fucomi
fucomi	%st(2)
fucomi	%st(2), %st

// CHECK: fnstsw %ax
// CHECK: fnstsw %ax

fnstsw
fnstsw %ax

// rdar://8431880
// CHECK: rclb	%bl
// CHECK: rcll	3735928559(%ebx,%ecx,8)
// CHECK: rcrl	%ecx
// CHECK: rcrl	305419896
rcl	%bl
rcll	0xdeadbeef(%ebx,%ecx,8)
rcr	%ecx
rcrl	0x12345678

rclb	%bl       // CHECK: rclb %bl     # encoding: [0xd0,0xd3]
rclb	$1, %bl   // CHECK: rclb %bl     # encoding: [0xd0,0xd3]
rclb	$2, %bl   // CHECK: rclb $2, %bl # encoding: [0xc0,0xd3,0x02]

// rdar://8418316
// PR12173
// CHECK: shldw	%cl, %bx, %dx
// CHECK: shldw	%cl, %bx, %dx
// CHECK: shldw	$1, %bx, %dx
// CHECK: shldw	%cl, %bx, (%rax)
// CHECK: shldw	%cl, %bx, (%rax)
// CHECK: shrdw	%cl, %bx, %dx
// CHECK: shrdw	%cl, %bx, %dx
// CHECK: shrdw	$1, %bx, %dx
// CHECK: shrdw	%cl, %bx, (%rax)
// CHECK: shrdw	%cl, %bx, (%rax)

shld  %bx, %dx
shld  %cl, %bx, %dx
shld  $1, %bx, %dx
shld  %bx, (%rax)
shld  %cl, %bx, (%rax)
shrd  %bx, %dx
shrd  %cl, %bx, %dx
shrd  $1, %bx, %dx
shrd  %bx, (%rax)
shrd  %cl, %bx, (%rax)

// CHECK: sldtl	%ecx
// CHECK: encoding: [0x0f,0x00,0xc1]
// CHECK: sldtw	%cx
// CHECK: encoding: [0x66,0x0f,0x00,0xc1]

sldt	%ecx
sldt	%cx

// CHECK: lcalll *3135175374 
// CHECK: ljmpl  *3135175374
// CHECK: lcalll *(%rax)
// CHECK: ljmpl *(%rax)
lcall  *0xbadeface
ljmp *0xbadeface
lcall *(%rax)
ljmpl *(%rax)

// CHECK: ljmpl *%cs:305419896
// CHECK:  encoding: [0x2e,0xff,0x2c,0x25,0x78,0x56,0x34,0x12]
ljmp %cs:*0x12345678

// rdar://8444631
// CHECK: enter	$31438, $0
// CHECK: encoding: [0xc8,0xce,0x7a,0x00]
// CHECK: enter	$31438, $1
// CHECK: encoding: [0xc8,0xce,0x7a,0x01]
// CHECK: enter	$31438, $127
// CHECK: encoding: [0xc8,0xce,0x7a,0x7f]
enter $0x7ace,$0
enter $0x7ace,$1
enter $0x7ace,$0x7f


// rdar://8456364
// CHECK: movw	%cs, %ax
mov %cs, %ax

// rdar://8456391
fcmovb %st(1), %st   // CHECK: fcmovb	%st(1), %st
fcmove %st(1), %st   // CHECK: fcmove	%st(1), %st
fcmovbe %st(1), %st  // CHECK: fcmovbe	%st(1), %st
fcmovu %st(1), %st   // CHECK: fcmovu	 %st(1), %st

fcmovnb %st(1), %st  // CHECK: fcmovnb	%st(1), %st
fcmovne %st(1), %st  // CHECK: fcmovne	%st(1), %st
fcmovnbe %st(1), %st // CHECK: fcmovnbe	%st(1), %st
fcmovnu %st(1), %st  // CHECK: fcmovnu	%st(1), %st

fcmovnae %st(1), %st // CHECK: fcmovb	%st(1), %st
fcmovna %st(1), %st  // CHECK: fcmovbe	%st(1), %st

fcmovae %st(1), %st  // CHECK: fcmovnb	%st(1), %st
fcmova %st(1), %st   // CHECK: fcmovnbe	%st(1), %st

// rdar://8456417
.byte (88 + 1) & 15  // CHECK: .byte	9

// rdar://8456412
mov %rdx, %cr0
// CHECK: movq	%rdx, %cr0
// CHECK: encoding: [0x0f,0x22,0xc2]
mov %rdx, %cr4
// CHECK: movq	%rdx, %cr4
// CHECK: encoding: [0x0f,0x22,0xe2]
mov %rdx, %cr8
// CHECK: movq	%rdx, %cr8
// CHECK: encoding: [0x44,0x0f,0x22,0xc2]
mov %rdx, %cr15
// CHECK: movq	%rdx, %cr15
// CHECK: encoding: [0x44,0x0f,0x22,0xfa]
mov %rdx, %dr15
// CHECK: movq	%rdx, %dr15
// CHECK: encoding: [0x44,0x0f,0x23,0xfa]
mov %rdx, %db15
// CHECK: movq	%rdx, %dr15
// CHECK: encoding: [0x44,0x0f,0x23,0xfa]

// rdar://8456371 - Handle commutable instructions written backward.
// CHECK: 	faddp	%st, %st(1)
// CHECK:	fmulp	%st, %st(2)
faddp %st, %st(1)
fmulp %st, %st(2)

// rdar://8468087 - Encode these accurately, they are not synonyms.
// CHECK: fmul	%st, %st(1)
// CHECK: encoding: [0xdc,0xc9]
// CHECK: fmul	%st(1)
// CHECK: encoding: [0xd8,0xc9]
fmul %st, %st(1)
fmul %st(1), %st

// CHECK: fadd	%st, %st(1)
// CHECK: encoding: [0xdc,0xc1]
// CHECK: fadd	%st(1)
// CHECK: encoding: [0xd8,0xc1]
fadd %st, %st(1)
fadd %st(1), %st


// rdar://8416805
// CHECK: xorb	%al, %al
// CHECK: encoding: [0x30,0xc0]
// CHECK: xorw	%di, %di
// CHECK: encoding: [0x66,0x31,0xff]
// CHECK: xorl	%esi, %esi
// CHECK: encoding: [0x31,0xf6]
// CHECK: xorq	%rsi, %rsi
// CHECK: encoding: [0x48,0x31,0xf6]
clrb    %al
clr    %di
clr    %esi
clr    %rsi

// rdar://8456378
cltq  // CHECK: cltq
cdqe  // CHECK: cltq
cwde  // CHECK: cwtl
cwtl  // CHECK: cwtl

// rdar://8416805
cbw   // CHECK: cbtw
cwd   // CHECK: cwtd
cdq   // CHECK: cltd
cqo   // CHECK: cqto

// rdar://8456378 and PR7557 - fstsw
fstsw %ax
// CHECK: wait
// CHECK: fnstsw
fstsw (%rax)
// CHECK: wait
// CHECK: fnstsw (%rax)

// PR8259
fstcw (%rsp)
// CHECK: wait
// CHECK: fnstcw (%rsp)

// PR8259
fstcw (%rsp)
// CHECK: wait
// CHECK: fnstcw (%rsp)

// PR8258
finit
// CHECK: wait
// CHECK: fninit

fsave	32493
// CHECK: wait
// CHECK: fnsave 32493


// rdar://8456382 - cvtsd2si support.
cvtsd2si	%xmm1, %rax
// CHECK: cvtsd2si	%xmm1, %rax
// CHECK: encoding: [0xf2,0x48,0x0f,0x2d,0xc1]
cvtsd2si	%xmm1, %eax
// CHECK: cvtsd2si	%xmm1, %eax
// CHECK: encoding: [0xf2,0x0f,0x2d,0xc1]

cvtsd2siq %xmm0, %rax // CHECK: cvtsd2si	%xmm0, %rax
cvtsd2sil %xmm0, %eax // CHECK: cvtsd2si	%xmm0, %eax
cvtsd2si %xmm0, %rax  // CHECK: cvtsd2si	%xmm0, %rax


cvttpd2dq %xmm1, %xmm0  // CHECK: cvttpd2dq %xmm1, %xmm0
cvttpd2dq (%rax), %xmm0 // CHECK: cvttpd2dq (%rax), %xmm0

cvttps2dq %xmm1, %xmm0  // CHECK: cvttps2dq %xmm1, %xmm0
cvttps2dq (%rax), %xmm0 // CHECK: cvttps2dq (%rax), %xmm0

// rdar://8456376 - llvm-mc rejects 'roundss'
roundss $0xE, %xmm0, %xmm0 // CHECK: encoding: [0x66,0x0f,0x3a,0x0a,0xc0,0x0e]
roundps $0xE, %xmm0, %xmm0 // CHECK: encoding: [0x66,0x0f,0x3a,0x08,0xc0,0x0e]
roundsd $0xE, %xmm0, %xmm0 // CHECK: encoding: [0x66,0x0f,0x3a,0x0b,0xc0,0x0e]
roundpd $0xE, %xmm0, %xmm0 // CHECK: encoding: [0x66,0x0f,0x3a,0x09,0xc0,0x0e]


// rdar://8482675 - 32-bit mem operand support in 64-bit mode (0x67 prefix)
leal	8(%eax), %esi
// CHECK: leal	8(%eax), %esi
// CHECK: encoding: [0x67,0x8d,0x70,0x08]
leaq	8(%eax), %rsi
// CHECK: leaq	8(%eax), %rsi
// CHECK: encoding: [0x67,0x48,0x8d,0x70,0x08]
leaq	8(%rax), %rsi
// CHECK: leaq	8(%rax), %rsi
// CHECK: encoding: [0x48,0x8d,0x70,0x08]


cvttpd2dq	0xdeadbeef(%ebx,%ecx,8),%xmm5
// CHECK: cvttpd2dq	3735928559(%ebx,%ecx,8), %xmm5
// CHECK: encoding: [0x67,0x66,0x0f,0xe6,0xac,0xcb,0xef,0xbe,0xad,0xde]

// rdar://8490728 - llvm-mc rejects 'movmskpd'
movmskpd	%xmm6, %rax
// CHECK: movmskpd	%xmm6, %eax
// CHECK: encoding: [0x66,0x0f,0x50,0xc6]
movmskpd	%xmm6, %eax
// CHECK: movmskpd	%xmm6, %eax
// CHECK: encoding: [0x66,0x0f,0x50,0xc6]

// rdar://8491845 - Gas supports commuted forms of non-commutable instructions.
fdivrp %st, %st(1) // CHECK: encoding: [0xde,0xf9]
fdivrp %st(1), %st // CHECK: encoding: [0xde,0xf9]

fsubrp %st, %st(1) // CHECK: encoding: [0xde,0xe9]
fsubrp %st(1), %st // CHECK: encoding: [0xde,0xe9]

// also PR8861
fdivp %st, %st(1) // CHECK: encoding: [0xde,0xf1]
fdivp %st(1), %st // CHECK: encoding: [0xde,0xf1]


movl	foo(%rip), %eax
// CHECK: movl	foo(%rip), %eax
// CHECK: encoding: [0x8b,0x05,A,A,A,A]
// CHECK: fixup A - offset: 2, value: foo-4, kind: reloc_riprel_4byte

movb	$12, foo(%rip)
// CHECK: movb	$12, foo(%rip)
// CHECK: encoding: [0xc6,0x05,A,A,A,A,0x0c]
// CHECK:    fixup A - offset: 2, value: foo-5, kind: reloc_riprel_4byte

movw	$12, foo(%rip)
// CHECK: movw	$12, foo(%rip)
// CHECK: encoding: [0x66,0xc7,0x05,A,A,A,A,0x0c,0x00]
// CHECK:    fixup A - offset: 3, value: foo-6, kind: reloc_riprel_4byte

movl	$12, foo(%rip)
// CHECK: movl	$12, foo(%rip)
// CHECK: encoding: [0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 2, value: foo-8, kind: reloc_riprel_4byte

// rdar://37247000
movl	$12, 1024(%rip)
// CHECK: movl	$12, 1024(%rip)
// CHECK: encoding: [0xc7,0x05,0x00,0x04,0x00,0x00,0x0c,0x00,0x00,0x00]

movq	$12, foo(%rip)
// CHECK:  movq	$12, foo(%rip)
// CHECK: encoding: [0x48,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 3, value: foo-8, kind: reloc_riprel_4byte

movl	foo(%eip), %eax
// CHECK: movl	foo(%eip), %eax
// CHECK: encoding: [0x67,0x8b,0x05,A,A,A,A]
// CHECK: fixup A - offset: 3, value: foo-4, kind: reloc_riprel_4byte

movb	$12, foo(%eip)
// CHECK: movb	$12, foo(%eip)
// CHECK: encoding: [0x67,0xc6,0x05,A,A,A,A,0x0c]
// CHECK:    fixup A - offset: 3, value: foo-5, kind: reloc_riprel_4byte

movw	$12, foo(%eip)
// CHECK: movw	$12, foo(%eip)
// CHECK: encoding: [0x67,0x66,0xc7,0x05,A,A,A,A,0x0c,0x00]
// CHECK:    fixup A - offset: 4, value: foo-6, kind: reloc_riprel_4byte

movl	$12, foo(%eip)
// CHECK: movl	$12, foo(%eip)
// CHECK: encoding: [0x67,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 3, value: foo-8, kind: reloc_riprel_4byte

movq	$12, foo(%eip)
// CHECK:  movq	$12, foo(%eip)
// CHECK: encoding: [0x67,0x48,0xc7,0x05,A,A,A,A,0x0c,0x00,0x00,0x00]
// CHECK:    fixup A - offset: 4, value: foo-8, kind: reloc_riprel_4byte

// CHECK: addq	$-424, %rax
// CHECK: encoding: [0x48,0x05,0x58,0xfe,0xff,0xff]
addq $-424, %rax


// CHECK: movq	_foo@GOTPCREL(%rip), %rax
// CHECK:  encoding: [0x48,0x8b,0x05,A,A,A,A]
// CHECK:  fixup A - offset: 3, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%rip), %rax

// CHECK: movq	_foo@GOTPCREL(%rip), %r14
// CHECK:  encoding: [0x4c,0x8b,0x35,A,A,A,A]
// CHECK:  fixup A - offset: 3, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%rip), %r14

// CHECK: movq	_foo@GOTPCREL(%eip), %rax
// CHECK:  encoding: [0x67,0x48,0x8b,0x05,A,A,A,A]
// CHECK:  fixup A - offset: 4, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%eip), %rax

// CHECK: movq	_foo@GOTPCREL(%eip), %r14
// CHECK:  encoding: [0x67,0x4c,0x8b,0x35,A,A,A,A]
// CHECK:  fixup A - offset: 4, value: _foo@GOTPCREL-4, kind: reloc_riprel_4byte_movq_load
movq _foo@GOTPCREL(%eip), %r14

// CHECK: movq	(%r13,%rax,8), %r13
// CHECK:  encoding: [0x4d,0x8b,0x6c,0xc5,0x00]
movq 0x00(%r13,%rax,8),%r13

// CHECK: testq	%rax, %rbx
// CHECK:  encoding: [0x48,0x85,0xc3]
testq %rax, %rbx

// CHECK: cmpq	%rbx, %r14
// CHECK:   encoding: [0x49,0x39,0xde]
        cmpq %rbx, %r14

// rdar://7947167

movsq
// CHECK: movsq
// CHECK:   encoding: [0x48,0xa5]

movsl
// CHECK: movsl
// CHECK:   encoding: [0xa5]

stosq
// CHECK: stosq
// CHECK:   encoding: [0x48,0xab]
stosl
// CHECK: stosl
// CHECK:   encoding: [0xab]


// Not moffset forms of moves, they are x86-32 only! rdar://7947184
movb	0, %al    // CHECK: movb 0, %al # encoding: [0x8a,0x04,0x25,0x00,0x00,0x00,0x00]
movw	0, %ax    // CHECK: movw 0, %ax # encoding: [0x66,0x8b,0x04,0x25,0x00,0x00,0x00,0x00]
movl	0, %eax   // CHECK: movl 0, %eax # encoding: [0x8b,0x04,0x25,0x00,0x00,0x00,0x00]

// CHECK: pushfq	# encoding: [0x9c]
        pushf
// CHECK: pushfq	# encoding: [0x9c]
        pushfq
// CHECK: popfq	        # encoding: [0x9d]
        popf
// CHECK: popfq	        # encoding: [0x9d]
        popfq

// CHECK: movabsq $-281474976710654, %rax
// CHECK: encoding: [0x48,0xb8,0x02,0x00,0x00,0x00,0x00,0x00,0xff,0xff]
        movabsq $0xFFFF000000000002, %rax

// CHECK: movabsq $-281474976710654, %rax
// CHECK: encoding: [0x48,0xb8,0x02,0x00,0x00,0x00,0x00,0x00,0xff,0xff]
        movq $0xFFFF000000000002, %rax

// CHECK: movabsq 81985529216486895, %rax
// CHECK: encoding: [0x48,0xa1,0xef,0xcd,0xab,0x89,0x67,0x45,0x23,0x01]
        movabsq 0x123456789abcdef, %rax

// CHECK: movq $-65536, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x00,0x00,0xff,0xff]
        movq $0xFFFFFFFFFFFF0000, %rax

// CHECK: movq $-256, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x00,0xff,0xff,0xff]
        movq $0xFFFFFFFFFFFFFF00, %rax

// CHECK: movq $10, %rax
// CHECK: encoding: [0x48,0xc7,0xc0,0x0a,0x00,0x00,0x00]
        movq $10, %rax

// CHECK: movq 81985529216486895, %rax
// CHECK: encoding: [0x48,0x8b,0x04,0x25,0xef,0xcd,0xab,0x89]
        movq 0x123456789abcdef, %rax

// CHECK: movabsb -6066930261531658096, %al
// CHECK: encoding: [0xa0,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsb 0xabcdef1234567890,%al

// CHECK: movabsw -6066930261531658096, %ax
// CHECK: encoding: [0x66,0xa1,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsw 0xabcdef1234567890,%ax

// CHECK: movabsl -6066930261531658096, %eax
// CHECK: encoding: [0xa1,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsl 0xabcdef1234567890,%eax

// CHECK: movabsq -6066930261531658096, %rax
// CHECK: encoding: [0x48,0xa1,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsq 0xabcdef1234567890, %rax

// CHECK: movabsb %al, -6066930261531658096
// CHECK: encoding: [0xa2,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsb %al,0xabcdef1234567890

// CHECK: movabsw %ax, -6066930261531658096
// CHECK: encoding: [0x66,0xa3,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsw %ax,0xabcdef1234567890

// CHECK: movabsl %eax, -6066930261531658096
// CHECK: encoding: [0xa3,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsl %eax,0xabcdef1234567890

// CHECK: movabsq %rax, -6066930261531658096
// CHECK: encoding: [0x48,0xa3,0x90,0x78,0x56,0x34,0x12,0xef,0xcd,0xab]
        movabsq %rax,0xabcdef1234567890

// rdar://8014869
//
// CHECK: ret
// CHECK:  encoding: [0xc3]
        retq

// CHECK: sete %al
// CHECK: encoding: [0x0f,0x94,0xc0]
        setz %al

// CHECK: setne %al
// CHECK: encoding: [0x0f,0x95,0xc0]
        setnz %al

// CHECK: je 0
// CHECK: encoding: [0x74,A]
        jz 0

// CHECK: jne
// CHECK: encoding: [0x75,A]
        jnz 0

// PR9264
btl	$1, 0 // CHECK: btl $1, 0 # encoding: [0x0f,0xba,0x24,0x25,0x00,0x00,0x00,0x00,0x01]
bt	$1, 0 // CHECK: btl $1, 0 # encoding: [0x0f,0xba,0x24,0x25,0x00,0x00,0x00,0x00,0x01]

// rdar://8017515
btq $0x01,%rdx
// CHECK: btq	$1, %rdx
// CHECK:  encoding: [0x48,0x0f,0xba,0xe2,0x01]

//rdar://8017633
// CHECK: movzbl	%al, %esi
// CHECK:  encoding: [0x0f,0xb6,0xf0]
        movzx %al, %esi

// CHECK: movzbq	%al, %rsi
// CHECK:  encoding: [0x48,0x0f,0xb6,0xf0]
        movzx %al, %rsi

// CHECK: movsbw	%al, %ax
// CHECK: encoding: [0x66,0x0f,0xbe,0xc0]
movsx %al, %ax

// CHECK: movsbl	%al, %eax
// CHECK: encoding: [0x0f,0xbe,0xc0]
movsx %al, %eax

// CHECK: movswl	%ax, %eax
// CHECK: encoding: [0x0f,0xbf,0xc0]
movsx %ax, %eax

// CHECK: movsbq	%bl, %rax
// CHECK: encoding: [0x48,0x0f,0xbe,0xc3]
movsx %bl, %rax

// CHECK: movswq %cx, %rax
// CHECK: encoding: [0x48,0x0f,0xbf,0xc1]
movsx %cx, %rax

// CHECK: movslq	%edi, %rax
// CHECK: encoding: [0x48,0x63,0xc7]
movsx %edi, %rax

// CHECK: movzbw	%al, %ax
// CHECK: encoding: [0x66,0x0f,0xb6,0xc0]
movzx %al, %ax

// CHECK: movzbl	%al, %eax
// CHECK: encoding: [0x0f,0xb6,0xc0]
movzx %al, %eax

// CHECK: movzwl	%ax, %eax
// CHECK: encoding: [0x0f,0xb7,0xc0]
movzx %ax, %eax

// CHECK: movzbq	%bl, %rax
// CHECK: encoding: [0x48,0x0f,0xb6,0xc3]
movzx %bl, %rax

// CHECK: movzwq	%cx, %rax
// CHECK: encoding: [0x48,0x0f,0xb7,0xc1]
movzx %cx, %rax

// CHECK: movsbw	(%rax), %ax
// CHECK: encoding: [0x66,0x0f,0xbe,0x00]
movsx (%rax), %ax

// CHECK: movzbw	(%rax), %ax
// CHECK: encoding: [0x66,0x0f,0xb6,0x00]
movzx (%rax), %ax

// CHECK: movzbq	1280(%rbx,%r11), %r12
// CHECK: encoding: [0x4e,0x0f,0xb6,0xa4,0x1b,0x00,0x05,0x00,0x00]
movzb 1280(%rbx, %r11), %r12

// rdar://7873482
// CHECK: [0x65,0x8b,0x04,0x25,0x7c,0x00,0x00,0x00]
        movl	%gs:124, %eax

// CHECK: jmpq *8(%rax)
// CHECK:   encoding: [0xff,0x60,0x08]
	jmp	*8(%rax)

// CHECK: btq $61, -216(%rbp)
// CHECK:   encoding: [0x48,0x0f,0xba,0xa5,0x28,0xff,0xff,0xff,0x3d]
	btq	$61, -216(%rbp)


// rdar://8061602
L1:
  jecxz L1
// CHECK: jecxz L1
// CHECK:   encoding: [0x67,0xe3,A]
  jrcxz L1
// CHECK: jrcxz L1
// CHECK:   encoding: [0xe3,A]

// PR8061
xchgl   368(%rax),%ecx
// CHECK: xchgl	%ecx, 368(%rax)
xchgl   %ecx, 368(%rax)
// CHECK: xchgl	%ecx, 368(%rax)

// rdar://8407548
xchg	0xdeadbeef(%rbx,%rcx,8),%bl
// CHECK: xchgb	%bl, 3735928559(%rbx,%rcx,8)



// PR7254
lock  incl 1(%rsp)
// CHECK: lock
// CHECK: incl 1(%rsp)

// rdar://8741045
lock/incl 1(%rsp)
// CHECK: lock
// CHECK: incl 1(%rsp)


lock addq %rsi, (%rdi)
// CHECK: lock
// CHECK: addq %rsi, (%rdi)
// CHECK: encoding: [0xf0,0x48,0x01,0x37]

lock subq %rsi, (%rdi)
// CHECK: lock
// CHECK: subq %rsi, (%rdi)
// CHECK: encoding: [0xf0,0x48,0x29,0x37]

lock andq %rsi, (%rdi)
// CHECK: lock
// CHECK: andq %rsi, (%rdi)
// CHECK: encoding: [0xf0,0x48,0x21,0x37]

lock orq %rsi, (%rdi)
// CHECK: lock
// CHECK: orq %rsi, (%rdi)
// CHECK: encoding: [0xf0,0x48,0x09,0x37]

lock xorq %rsi, (%rdi)
// CHECK: lock
// CHECK: xorq %rsi, (%rdi)
// CHECK: encoding: [0xf0,0x48,0x31,0x37]

xacquire lock addq %rax, (%rax)
// CHECK: xacquire
// CHECK: encoding: [0xf2]
// CHECK: lock
// CHECK: addq %rax, (%rax)
// CHECK: encoding: [0xf0,0x48,0x01,0x00]

xrelease lock addq %rax, (%rax)
// CHECK: xrelease
// CHECK: encoding: [0xf3]
// CHECK: lock
// CHECK: addq %rax, (%rax)
// CHECK: encoding: [0xf0,0x48,0x01,0x00]

// rdar://8033482
rep movsl
// CHECK: rep
// CHECK: movsl
// CHECK: encoding: [0xf3,0xa5]


// rdar://8403974
iret
// CHECK: iretl
// CHECK: encoding: [0xcf]
iretw
// CHECK: iretw
// CHECK: encoding: [0x66,0xcf]
iretl
// CHECK: iretl
// CHECK: encoding: [0xcf]
iretq
// CHECK: iretq
// CHECK: encoding: [0x48,0xcf]

// rdar://8416805
// CHECK: retw	$31438
// CHECK:  encoding: [0x66,0xc2,0xce,0x7a]
        	retw	$0x7ace

// CHECK: lretw	$31438
// CHECK:  encoding: [0x66,0xca,0xce,0x7a]
        	lretw	$0x7ace

// PR8592
lretq  // CHECK: lretq # encoding: [0x48,0xcb]
lretl  // CHECK: lretl # encoding: [0xcb]
lret   // CHECK: lretl # encoding: [0xcb]
lretw  // CHECK: lretw # encoding: [0x66,0xcb]

// rdar://8403907
sysret
// CHECK: sysretl
// CHECK: encoding: [0x0f,0x07]
sysretl
// CHECK: sysretl
// CHECK: encoding: [0x0f,0x07]
sysretq
// CHECK: sysretq
// CHECK: encoding: [0x48,0x0f,0x07]

// rdar://8407242
push %fs
// CHECK: pushq	%fs
// CHECK: encoding: [0x0f,0xa0]
push %gs
// CHECK: pushq	%gs
// CHECK: encoding: [0x0f,0xa8]

pushw %fs
// CHECK: pushw	%fs
// CHECK: encoding: [0x66,0x0f,0xa0]
pushw %gs
// CHECK: pushw	%gs
// CHECK: encoding: [0x66,0x0f,0xa8]


pop %fs
// CHECK: popq	%fs
// CHECK: encoding: [0x0f,0xa1]
pop %gs
// CHECK: popq	%gs
// CHECK: encoding: [0x0f,0xa9]

popw %fs
// CHECK: popw	%fs
// CHECK: encoding: [0x66,0x0f,0xa1]
popw %gs
// CHECK: popw	%gs
// CHECK: encoding: [0x66,0x0f,0xa9]

// rdar://8438816
fildq -8(%rsp)
fildll -8(%rsp)
// CHECK: fildll	-8(%rsp)
// CHECK: encoding: [0xdf,0x6c,0x24,0xf8]
// CHECK: fildll	-8(%rsp)
// CHECK: encoding: [0xdf,0x6c,0x24,0xf8]

// CHECK: callq a
        callq a

// CHECK: leaq	-40(%rbp), %r15
	leaq	-40(%rbp), %r15



// rdar://8013734 - Alias dr6=db6
mov %dr6, %rax
mov %db6, %rax
// CHECK: movq	%dr6, %rax
// CHECK: movq	%dr6, %rax


// INC/DEC encodings.
incb %al  // CHECK:	incb	%al # encoding: [0xfe,0xc0]
incw %ax  // CHECK:	incw	%ax # encoding: [0x66,0xff,0xc0]
incl %eax // CHECK:	incl	%eax # encoding: [0xff,0xc0]
decb %al  // CHECK:	decb	%al # encoding: [0xfe,0xc8]
decw %ax  // CHECK:	decw	%ax # encoding: [0x66,0xff,0xc8]
decl %eax // CHECK:	decl	%eax # encoding: [0xff,0xc8]

// rdar://8416805
// CHECK: lgdtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x50,0x04]
        	lgdt	4(%rax)

// CHECK: lgdtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x50,0x04]
        	lgdtq	4(%rax)

// CHECK: lidtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x58,0x04]
        	lidt	4(%rax)

// CHECK: lidtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x58,0x04]
        	lidtq	4(%rax)

// CHECK: sgdtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x40,0x04]
        	sgdt	4(%rax)

// CHECK: sgdtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x40,0x04]
        	sgdtq	4(%rax)

// CHECK: sidtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x48,0x04]
        	sidt	4(%rax)

// CHECK: sidtq	4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x48,0x04]
        	sidtq	4(%rax)


// rdar://8208615
mov (%rsi), %gs  // CHECK: movw	(%rsi), %gs # encoding: [0x8e,0x2e]
mov %gs, (%rsi)  // CHECK: movw	%gs, (%rsi) # encoding: [0x8c,0x2e]


// rdar://8431864
//CHECK: divb	%bl
//CHECK: divw	%bx
//CHECK: divl	%ecx
//CHECK: divl	3735928559(%ebx,%ecx,8)
//CHECK: divl	69
//CHECK: divl	32493
//CHECK: divl	3133065982
//CHECK: divl	305419896
//CHECK: idivb	%bl
//CHECK: idivw	%bx
//CHECK: idivl	%ecx
//CHECK: idivl	3735928559(%ebx,%ecx,8)
//CHECK: idivl	69
//CHECK: idivl	32493
//CHECK: idivl	3133065982
//CHECK: idivl	305419896
	div	%bl,%al
	div	%bx,%ax
	div	%ecx,%eax
	div	0xdeadbeef(%ebx,%ecx,8),%eax
	div	0x45,%eax
	div	0x7eed,%eax
	div	0xbabecafe,%eax
	div	0x12345678,%eax
	idiv	%bl,%al
	idiv	%bx,%ax
	idiv	%ecx,%eax
	idiv	0xdeadbeef(%ebx,%ecx,8),%eax
	idiv	0x45,%eax
	idiv	0x7eed,%eax
	idiv	0xbabecafe,%eax
	idiv	0x12345678,%eax

// PR8524
movd	%rax, %mm5 // CHECK: movq %rax, %mm5 # encoding: [0x48,0x0f,0x6e,0xe8]
movd	%mm5, %rbx // CHECK: movq %mm5, %rbx # encoding: [0x48,0x0f,0x7e,0xeb]
movq	%rax, %mm5 // CHECK: movq %rax, %mm5 # encoding: [0x48,0x0f,0x6e,0xe8]
movq	%mm5, %rbx // CHECK: movq %mm5, %rbx # encoding: [0x48,0x0f,0x7e,0xeb]

rex64 // CHECK: rex64 # encoding: [0x48]
data16 // CHECK: data16 # encoding: [0x66]

// CHECK: data16
// CHECK: encoding: [0x66]
// CHECK: lgdtq 4(%rax)
// CHECK:  encoding: [0x0f,0x01,0x50,0x04]
data16 lgdt 4(%rax)

// PR8855
movq 18446744073709551615,%rbx   // CHECK: movq	-1, %rbx

// PR8946
movdqu	%xmm0, %xmm1 // CHECK: movdqu	%xmm0, %xmm1 # encoding: [0xf3,0x0f,0x6f,0xc8]

// PR8935
xgetbv // CHECK: xgetbv # encoding: [0x0f,0x01,0xd0]
xsetbv // CHECK: xsetbv # encoding: [0x0f,0x01,0xd1]

// CHECK: loope 0
// CHECK: encoding: [0xe1,A]
	loopz 0

// CHECK: loopne 0
// CHECK: encoding: [0xe0,A]
	loopnz 0

// CHECK: outsb (%rsi), %dx # encoding: [0x6e]
// CHECK: outsb
// CHECK: outsb
	outsb
	outsb	%ds:(%rsi), %dx
	outsb	(%rsi), %dx

// CHECK: outsw (%rsi), %dx # encoding: [0x66,0x6f]
// CHECK: outsw
// CHECK: outsw
	outsw
	outsw	%ds:(%rsi), %dx
	outsw	(%rsi), %dx

// CHECK: outsl (%rsi), %dx # encoding: [0x6f]
// CHECK: outsl
	outsl
	outsl	%ds:(%rsi), %dx
	outsl	(%rsi), %dx

// CHECK: insb  %dx, %es:(%rdi) # encoding: [0x6c]
// CHECK: insb
	insb
	insb	%dx, %es:(%rdi)

// CHECK: insw  %dx, %es:(%rdi) # encoding: [0x66,0x6d]
// CHECK: insw
	insw
	insw	%dx, %es:(%rdi)

// CHECK: insl %dx, %es:(%rdi) # encoding: [0x6d]
// CHECK: insl
	insl
	insl	%dx, %es:(%rdi)

// CHECK: movsb (%rsi), %es:(%rdi) # encoding: [0xa4]
// CHECK: movsb
// CHECK: movsb
	movsb
	movsb	%ds:(%rsi), %es:(%rdi)
	movsb	(%rsi), %es:(%rdi)

// CHECK: movsw (%rsi), %es:(%rdi) # encoding: [0x66,0xa5]
// CHECK: movsw
// CHECK: movsw
	movsw
	movsw	%ds:(%rsi), %es:(%rdi)
	movsw	(%rsi), %es:(%rdi)

// CHECK: movsl (%rsi), %es:(%rdi) # encoding: [0xa5]
// CHECK: movsl
// CHECK: movsl
	movsl
	movsl	%ds:(%rsi), %es:(%rdi)
	movsl	(%rsi), %es:(%rdi)
// rdar://10883092
// CHECK: movsl
	movsl	(%rsi), (%rdi)

// CHECK: movsq (%rsi), %es:(%rdi) # encoding: [0x48,0xa5]
// CHECK: movsq
// CHECK: movsq
	movsq
	movsq	%ds:(%rsi), %es:(%rdi)
	movsq	(%rsi), %es:(%rdi)

// CHECK: lodsb (%rsi), %al # encoding: [0xac]
// CHECK: lodsb
// CHECK: lodsb
// CHECK: lodsb
// CHECK: lodsb
	lodsb
	lodsb	%ds:(%rsi), %al
	lodsb	(%rsi), %al
	lods	%ds:(%rsi), %al
	lods	(%rsi), %al

// CHECK: lodsw (%rsi), %ax # encoding: [0x66,0xad]
// CHECK: lodsw
// CHECK: lodsw
// CHECK: lodsw
// CHECK: lodsw
	lodsw
	lodsw	%ds:(%rsi), %ax
	lodsw	(%rsi), %ax
	lods	%ds:(%rsi), %ax
	lods	(%rsi), %ax

// CHECK: lodsl (%rsi), %eax # encoding: [0xad]
// CHECK: lodsl
// CHECK: lodsl
// CHECK: lodsl
// CHECK: lodsl
	lodsl
	lodsl	%ds:(%rsi), %eax
	lodsl	(%rsi), %eax
	lods	%ds:(%rsi), %eax
	lods	(%rsi), %eax

// CHECK: lodsq (%rsi), %rax # encoding: [0x48,0xad]
// CHECK: lodsq
// CHECK: lodsq
// CHECK: lodsq
// CHECK: lodsq
	lodsq
	lodsq	%ds:(%rsi), %rax
	lodsq	(%rsi), %rax
	lods	%ds:(%rsi), %rax
	lods	(%rsi), %rax

// CHECK: stosb %al, %es:(%rdi) # encoding: [0xaa]
// CHECK: stosb
// CHECK: stosb
	stosb
	stosb	%al, %es:(%rdi)
	stos	%al, %es:(%rdi)

// CHECK: stosw %ax, %es:(%rdi) # encoding: [0x66,0xab]
// CHECK: stosw
// CHECK: stosw
	stosw
	stosw	%ax, %es:(%rdi)
	stos	%ax, %es:(%rdi)

// CHECK: stosl %eax, %es:(%rdi) # encoding: [0xab]
// CHECK: stosl
// CHECK: stosl
	stosl
	stosl	%eax, %es:(%rdi)
	stos	%eax, %es:(%rdi)

// CHECK: stosq %rax, %es:(%rdi) # encoding: [0x48,0xab]
// CHECK: stosq
// CHECK: stosq
	stosq
	stosq	%rax, %es:(%rdi)
	stos	%rax, %es:(%rdi)

// CHECK: strw
// CHECK: encoding: [0x66,0x0f,0x00,0xc8]
	str %ax

// CHECK: strl
// CHECK: encoding: [0x0f,0x00,0xc8]
	str %eax

// CHECK: strw
// CHECK: encoding: [0x66,0x0f,0x00,0xc8]
	str %ax

// CHECK: strq
// CHECK: encoding: [0x48,0x0f,0x00,0xc8]
	str %rax

// CHECK: movq %rdi, %xmm0
// CHECK: encoding: [0x66,0x48,0x0f,0x6e,0xc7]
	movq %rdi,%xmm0

// CHECK: movq  %xmm0, %rax
// CHECK: encoding: [0x66,0x48,0x0f,0x7e,0xc0]
    movq  %xmm0, %rax

// CHECK: movntil %eax, (%rdi)
// CHECK: encoding: [0x0f,0xc3,0x07]
// CHECK: movntil
movntil %eax, (%rdi)
movnti %eax, (%rdi)

// CHECK: movntiq %rax, (%rdi)
// CHECK: encoding: [0x48,0x0f,0xc3,0x07]
// CHECK: movntiq
movntiq %rax, (%rdi)
movnti %rax, (%rdi)

// CHECK: pclmulqdq	$17, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0x44,0xc8,0x11]
pclmulhqhqdq %xmm0, %xmm1

// CHECK: pclmulqdq	$1, %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0x44,0xc8,0x01]
pclmulqdq $1, %xmm0, %xmm1

// CHECK: pclmulqdq	$16, (%rdi), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0x44,0x0f,0x10]
pclmullqhqdq (%rdi), %xmm1

// CHECK: pclmulqdq	$0, (%rdi), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0x44,0x0f,0x00]
pclmulqdq $0, (%rdi), %xmm1

// PR10345
// CHECK: nop
// CHECK: encoding: [0x90]
xchgq %rax, %rax

// CHECK: xchgl %eax, %eax
// CHECK: encoding: [0x87,0xc0]
xchgl %eax, %eax

// CHECK: xchgw %ax, %ax
// CHECK: encoding: [0x66,0x90]
xchgw %ax, %ax

// CHECK: xchgl %ecx, %eax
// CHECK: encoding: [0x91]
xchgl %ecx, %eax

// CHECK: xchgl %ecx, %eax
// CHECK: encoding: [0x91]
xchgl %eax, %ecx

// CHECK: sysexit
// CHECK: encoding: [0x0f,0x35]
sysexit

// CHECK: sysexitl
// CHECK: encoding: [0x0f,0x35]
sysexitl

// CHECK: sysexitq
// CHECK: encoding: [0x48,0x0f,0x35]
sysexitq

// CHECK: clac
// CHECK: encoding: [0x0f,0x01,0xca]
clac

// CHECK: stac
// CHECK: encoding: [0x0f,0x01,0xcb]
stac

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
faddp %st, %st(1)
fmulp %st, %st(1)
fsubp %st, %st(1)
fsubrp %st, %st(1)
fdivp %st, %st(1)
fdivrp %st, %st(1)

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
faddp %st(1), %st
fmulp %st(1), %st
fsubp %st(1), %st
fsubrp %st(1), %st
fdivp %st(1), %st
fdivrp %st(1), %st

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
faddp %st(1)
fmulp %st(1)
fsubp %st(1)
fsubrp %st(1)
fdivp %st(1)
fdivrp %st(1)

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
faddp
fmulp
fsubp
fsubrp
fdivp
fdivrp

// CHECK: fadd %st(1)
// CHECK: fmul %st(1)
// CHECK: fsub %st(1)
// CHECK: fsubr %st(1)
// CHECK: fdiv %st(1)
// CHECK: fdivr %st(1)
fadd %st(1), %st
fmul %st(1), %st
fsub %st(1), %st
fsubr %st(1), %st
fdiv %st(1), %st
fdivr %st(1), %st

// CHECK: fadd %st, %st(1)
// CHECK: fmul %st, %st(1)
// CHECK: fsub %st, %st(1)
// CHECK: fsubr %st, %st(1)
// CHECK: fdiv %st, %st(1)
// CHECK: fdivr %st, %st(1)
fadd %st, %st(1)
fmul %st, %st(1)
fsub %st, %st(1)
fsubr %st, %st(1)
fdiv %st, %st(1)
fdivr %st, %st(1)

// CHECK: fadd %st(1)
// CHECK: fmul %st(1)
// CHECK: fsub %st(1)
// CHECK: fsubr %st(1)
// CHECK: fdiv %st(1)
// CHECK: fdivr %st(1)
fadd %st(1)
fmul %st(1)
fsub %st(1)
fsubr %st(1)
fdiv %st(1)
fdivr %st(1)

// CHECK: movd %xmm0, %eax
// CHECK: movq %xmm0, %rax
// CHECK: movq %xmm0, %rax
// CHECK: vmovd %xmm0, %eax
// CHECK: vmovq %xmm0, %rax
// CHECK: vmovq %xmm0, %rax
movd %xmm0, %eax
movq %xmm0, %rax
movq %xmm0, %rax
vmovd %xmm0, %eax
vmovd %xmm0, %rax
vmovq %xmm0, %rax

// CHECK: seto 3735928559(%r10,%r9,8)
// CHECK:  encoding: [0x43,0x0f,0x90,0x84,0xca,0xef,0xbe,0xad,0xde]
	seto 0xdeadbeef(%r10,%r9,8)

// CHECK: 	monitorx
// CHECK:  encoding: [0x0f,0x01,0xfa]
        	monitorx

// CHECK: 	monitorx
// CHECK:  encoding: [0x0f,0x01,0xfa]
        	monitorx %rax, %rcx, %rdx

// CHECK: 	mwaitx
// CHECK:  encoding: [0x0f,0x01,0xfb]
        	mwaitx

// CHECK: 	mwaitx
// CHECK:  encoding: [0x0f,0x01,0xfb]
        	mwaitx %rax, %rcx, %rbx

// CHECK:       clzero
// CHECK:  encoding: [0x0f,0x01,0xfc]
                clzero

// CHECK:       clzero
// CHECK:  encoding: [0x0f,0x01,0xfc]
                clzero %rax

// CHECK: 	movl %r15d, (%r15,%r15)
// CHECK:  encoding: [0x47,0x89,0x3c,0x3f]
movl %r15d, (%r15,%r15)

// CHECK: nopq	3735928559(%rbx,%rcx,8)
// CHECK:  encoding: [0x48,0x0f,0x1f,0x84,0xcb,0xef,0xbe,0xad,0xde]
nopq	0xdeadbeef(%rbx,%rcx,8)

// CHECK: nopq	%rax
// CHECK:  encoding: [0x48,0x0f,0x1f,0xc0]
nopq	%rax

// CHECK: rdpid %rax
// CHECK: encoding: [0xf3,0x0f,0xc7,0xf8]
rdpid %rax

// CHECK: ptwritel 3735928559(%rbx,%rcx,8)
// CHECK:  encoding: [0xf3,0x0f,0xae,0xa4,0xcb,0xef,0xbe,0xad,0xde]
ptwritel 0xdeadbeef(%rbx,%rcx,8)

// CHECK: ptwritel %eax
// CHECK:  encoding: [0xf3,0x0f,0xae,0xe0]
ptwritel %eax

// CHECK: ptwriteq 3735928559(%rbx,%rcx,8)
// CHECK:  encoding: [0xf3,0x48,0x0f,0xae,0xa4,0xcb,0xef,0xbe,0xad,0xde]
ptwriteq 0xdeadbeef(%rbx,%rcx,8)

// CHECK: ptwriteq %rax
// CHECK:  encoding: [0xf3,0x48,0x0f,0xae,0xe0]
ptwriteq %rax

// CHECK: wbnoinvd
// CHECK:  encoding: [0xf3,0x0f,0x09]
wbnoinvd

// CHECK: cldemote 4(%rax)
// CHECK:  encoding: [0x0f,0x1c,0x40,0x04]
cldemote 4(%rax)

// CHECK: cldemote 3735928559(%rbx,%rcx,8)
// CHECK:  encoding: [0x0f,0x1c,0x84,0xcb,0xef,0xbe,0xad,0xde]
cldemote 0xdeadbeef(%rbx,%rcx,8)

// CHECK: umonitor %r13
// CHECK:  encoding: [0xf3,0x41,0x0f,0xae,0xf5]
umonitor %r13

// CHECK: umonitor %rax
// CHECK:  encoding: [0xf3,0x0f,0xae,0xf0]
umonitor %rax

// CHECK: umonitor %eax
// CHECK:  encoding: [0x67,0xf3,0x0f,0xae,0xf0]
umonitor %eax

// CHECK: umwait %r15
// CHECK:  encoding: [0xf2,0x41,0x0f,0xae,0xf7]
umwait %r15

// CHECK: umwait %ebx
// CHECK:  encoding: [0xf2,0x0f,0xae,0xf3]
umwait %ebx

// CHECK: tpause %r15
// CHECK:  encoding: [0x66,0x41,0x0f,0xae,0xf7]
tpause %r15

// CHECK: tpause %ebx
// CHECK:  encoding: [0x66,0x0f,0xae,0xf3]
tpause %ebx

// CHECK: movdiri %r15, 485498096
// CHECK: # encoding: [0x4c,0x0f,0x38,0xf9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
movdiri %r15, 485498096

// CHECK: movdiri %r15, (%rdx)
// CHECK: # encoding: [0x4c,0x0f,0x38,0xf9,0x3a]
movdiri %r15, (%rdx)

// CHECK: movdiri %r15, 64(%rdx)
// CHECK: # encoding: [0x4c,0x0f,0x38,0xf9,0x7a,0x40]
movdiri %r15, 64(%rdx)

// CHECK: movdir64b 485498096, %rax
// CHECK: # encoding: [0x66,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
movdir64b 485498096, %rax

// CHECK: movdir64b 485498096, %eax
// CHECK: # encoding: [0x67,0x66,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
movdir64b 485498096, %eax

// CHECK: movdir64b (%rdx), %r15
// CHECK: # encoding: [0x66,0x44,0x0f,0x38,0xf8,0x3a]
movdir64b (%rdx), %r15

// CHECK: pconfig
// CHECK: # encoding: [0x0f,0x01,0xc5]
pconfig

// CHECK: encls
// CHECK: encoding: [0x0f,0x01,0xcf]
encls

// CHECK: enclu
// CHECK: encoding: [0x0f,0x01,0xd7]
enclu

// CHECK: enclv
// CHECK: encoding: [0x0f,0x01,0xc0]
enclv

// CHECK: movq %rax, %rbx
// CHECK: encoding: [0x48,0x8b,0xd8]
movq.s %rax, %rbx

// CHECK: movq %rax, %rbx
// CHECK: encoding: [0x48,0x8b,0xd8]
mov.s %rax, %rbx

// CHECK: movl %eax, %ebx
// CHECK: encoding: [0x8b,0xd8]
movl.s %eax, %ebx

// CHECK: movl %eax, %ebx
// CHECK: encoding: [0x8b,0xd8]
mov.s %eax, %ebx

// CHECK: movw %ax, %bx
// CHECK: encoding: [0x66,0x8b,0xd8]
movw.s %ax, %bx

// CHECK: movw %ax, %bx
// CHECK: encoding: [0x66,0x8b,0xd8]
mov.s %ax, %bx

// CHECK: movb %al, %bl
// CHECK: encoding: [0x8a,0xd8]
movb.s %al, %bl

// CHECK: movb %al, %bl
// CHECK: encoding: [0x8a,0xd8]
mov.s %al, %bl

// CHECK: movq %mm0, %mm1
// CHECK: encoding: [0x0f,0x7f,0xc1]
movq.s %mm0, %mm1

// CHECK: movq %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0xd6,0xc1]
movq.s %xmm0, %xmm1

// CHECK: movdqa %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0x7f,0xc1]
movdqa.s %xmm0, %xmm1

// CHECK: movdqu %xmm0, %xmm1
// CHECK: encoding: [0xf3,0x0f,0x7f,0xc1]
movdqu.s %xmm0, %xmm1

// CHECK: movaps %xmm0, %xmm1
// CHECK: encoding: [0x0f,0x29,0xc1]
movaps.s %xmm0, %xmm1

// CHECK: movups %xmm0, %xmm1
// CHECK: encoding: [0x0f,0x11,0xc1]
movups.s %xmm0, %xmm1

// CHECK: movapd %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0x29,0xc1]
movapd.s %xmm0, %xmm1

// CHECK: movupd %xmm0, %xmm1
// CHECK: encoding: [0x66,0x0f,0x11,0xc1]
movupd.s %xmm0, %xmm1

// CHECK: vmovq %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x79,0xd6,0xc0]
vmovq.s %xmm0, %xmm8

// CHECK: vmovq %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0xd6,0xc0]
vmovq.s %xmm8, %xmm0

// CHECK: vmovdqa %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x79,0x7f,0xc0]
vmovdqa.s %xmm0, %xmm8

// CHECK: vmovdqa %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0x7f,0xc0]
vmovdqa.s %xmm8, %xmm0

// CHECK: vmovdqu %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x7a,0x7f,0xc0]
vmovdqu.s %xmm0, %xmm8

// CHECK: vmovdqu %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x7a,0x7f,0xc0]
vmovdqu.s %xmm8, %xmm0

// CHECK: vmovaps %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x78,0x29,0xc0]
vmovaps.s %xmm0, %xmm8

// CHECK: vmovaps %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x78,0x29,0xc0]
vmovaps.s %xmm8, %xmm0

// CHECK: vmovups %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x78,0x11,0xc0]
vmovups.s %xmm0, %xmm8

// CHECK: vmovups %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x78,0x11,0xc0]
vmovups.s %xmm8, %xmm0

// CHECK: vmovapd %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x79,0x29,0xc0]
vmovapd.s %xmm0, %xmm8

// CHECK: vmovapd %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0x29,0xc0]
vmovapd.s %xmm8, %xmm0

// CHECK: vmovupd %xmm0, %xmm8
// CHECK: encoding: [0xc4,0xc1,0x79,0x11,0xc0]
vmovupd.s %xmm0, %xmm8

// CHECK: vmovupd %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0x11,0xc0]
vmovupd.s %xmm8, %xmm0

//  __asm __volatile(
//    "pushf        \n\t"
//    "popf       \n\t"
//    "rep        \n\t"
//    ".byte  0x0f, 0xa7, 0xd0"
//  );
// CHECK: pushfq
// CHECK-NEXT: popfq
// CHECK-NEXT: rep
// CHECK-NEXT: .byte 15
// CHECK-NEXT: .byte 167
// CHECK-NEXT: .byte 208
pushfq
popfq
rep
.byte 15
.byte 167
.byte 208

// CHECK: lock
// CHECK: cmpxchgl
        cmp $0, %edx
        je 1f
        lock
1:      cmpxchgl %ecx,(%rdi)

// CHECK: rep
// CHECK-NEXT: byte
rep
.byte 0xa4      # movsb

// CHECK: lock
// This line has to be the last one in the file
lock

// CHECK: enqcmd 268435456(%ebp,%r14d,8), %esi
// CHECK: encoding: [0x67,0xf2,0x42,0x0f,0x38,0xf8,0xb4,0xf5,0x00,0x00,0x00,0x10]
enqcmd  0x10000000(%ebp, %r14d, 8), %esi

// CHECK: enqcmd (%r9d), %edi
// CHECK: encoding: [0x67,0xf2,0x41,0x0f,0x38,0xf8,0x39]
enqcmd  (%r9d), %edi

// CHECK: enqcmd 8128(%ecx), %eax
// CHECK: encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x81,0xc0,0x1f,0x00,0x00]
enqcmd  8128(%ecx), %eax

// CHECK: enqcmd -8192(%edx), %ebx
// CHECK: encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x9a,0x00,0xe0,0xff,0xff]
enqcmd  -8192(%edx), %ebx

// CHECK: enqcmd 485498096, %eax
// CHECK: encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
enqcmd 485498096, %eax

// CHECK: enqcmds 268435456(%ebp,%r14d,8), %esi
// CHECK: encoding: [0x67,0xf3,0x42,0x0f,0x38,0xf8,0xb4,0xf5,0x00,0x00,0x00,0x10]
enqcmds 0x10000000(%ebp, %r14d, 8), %esi

// CHECK: enqcmds (%r9d), %edi
// CHECK: encoding: [0x67,0xf3,0x41,0x0f,0x38,0xf8,0x39]
enqcmds (%r9d), %edi

// CHECK: enqcmds 8128(%ecx), %eax
// CHECK: encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x81,0xc0,0x1f,0x00,0x00]
enqcmds 8128(%ecx), %eax

// CHECK: enqcmds -8192(%edx), %ebx
// CHECK: encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x9a,0x00,0xe0,0xff,0xff]
enqcmds -8192(%edx), %ebx

// CHECK: enqcmds 485498096, %eax
// CHECK: encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
enqcmds 485498096, %eax

// CHECK: enqcmd 268435456(%rbp,%r14,8), %rsi
// CHECK: encoding: [0xf2,0x42,0x0f,0x38,0xf8,0xb4,0xf5,0x00,0x00,0x00,0x10]
enqcmd  0x10000000(%rbp, %r14, 8), %rsi

// CHECK: enqcmd (%r9), %rdi
// CHECK: encoding: [0xf2,0x41,0x0f,0x38,0xf8,0x39]
enqcmd  (%r9), %rdi

// CHECK: enqcmd 8128(%rcx), %rax
// CHECK: encoding: [0xf2,0x0f,0x38,0xf8,0x81,0xc0,0x1f,0x00,0x00]
enqcmd  8128(%rcx), %rax

// CHECK: enqcmd -8192(%rdx), %rbx
// CHECK: encoding: [0xf2,0x0f,0x38,0xf8,0x9a,0x00,0xe0,0xff,0xff]
enqcmd  -8192(%rdx), %rbx

// CHECK: enqcmd 485498096, %rax
// CHECK: encoding: [0xf2,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
enqcmd 485498096, %rax

// CHECK: enqcmds 268435456(%rbp,%r14,8), %rsi
// CHECK: encoding: [0xf3,0x42,0x0f,0x38,0xf8,0xb4,0xf5,0x00,0x00,0x00,0x10]
enqcmds 0x10000000(%rbp, %r14, 8), %rsi

// CHECK: enqcmds (%r9), %rdi
// CHECK: encoding: [0xf3,0x41,0x0f,0x38,0xf8,0x39]
enqcmds (%r9), %rdi

// CHECK: enqcmds 8128(%rcx), %rax
// CHECK: encoding: [0xf3,0x0f,0x38,0xf8,0x81,0xc0,0x1f,0x00,0x00]
enqcmds 8128(%rcx), %rax

// CHECK: enqcmds -8192(%rdx), %rbx
// CHECK: encoding: [0xf3,0x0f,0x38,0xf8,0x9a,0x00,0xe0,0xff,0xff]
enqcmds -8192(%rdx), %rbx

// CHECK: enqcmds 485498096, %rax
// CHECK: encoding: [0xf3,0x0f,0x38,0xf8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]
enqcmds 485498096, %rax

// CHECK: serialize
// CHECK: encoding: [0x0f,0x01,0xe8]
serialize

// CHECK: xsusldtrk
// CHECK: encoding: [0xf2,0x0f,0x01,0xe8]
xsusldtrk

// CHECK: xresldtrk
// CHECK: encoding: [0xf2,0x0f,0x01,0xe9]
xresldtrk

// CHECK: ud1q %rdx, %rdi
// CHECK:  encoding: [0x48,0x0f,0xb9,0xfa]
ud1 %rdx, %rdi

// CHECK: ud1q (%rbx), %rcx
// CHECK:  encoding: [0x48,0x0f,0xb9,0x0b]
ud2b (%rbx), %rcx

// Requires no displacement by default
// CHECK: movl $1, (%rax)
// CHECK: encoding: [0xc7,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax)
// CHECK: encoding: [0xc7,0x40,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax)
// CHECK: encoding: [0xc7,0x80,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax)
// CHECK: encoding: [0xc7,0x40,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax)
// CHECK: encoding: [0xc7,0x80,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%rax)
{disp8} movl $1, (%rax)
{disp32} movl $1, (%rax)
movl.d8 $1, (%rax)
movl.d32 $1, (%rax)

// Requires disp8 by default
// CHECK: movl $1, (%rbp)
// CHECK: encoding: [0xc7,0x45,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rbp)
// CHECK: encoding: [0xc7,0x45,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rbp)
// CHECK: encoding: [0xc7,0x85,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%rbp)
{disp8} movl $1, (%rbp)
{disp32} movl $1, (%rbp)

// Requires disp8 by default
// CHECK: movl $1, (%r13)
// CHECK: encoding: [0x41,0xc7,0x45,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%r13)
// CHECK: encoding: [0x41,0xc7,0x45,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%r13)
// CHECK: encoding: [0x41,0xc7,0x85,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%r13)
{disp8} movl $1, (%r13)
{disp32} movl $1, (%r13)

// Requires disp8 by default
// CHECK: movl $1, 8(%rax)
// CHECK: encoding: [0xc7,0x40,0x08,0x01,0x00,0x00,0x00]
// CHECK: movl $1, 8(%rax)
// CHECK: encoding: [0xc7,0x40,0x08,0x01,0x00,0x00,0x00]
// CHECK: movl $1, 8(%rax)
// CHECK: encoding: [0xc7,0x80,0x08,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, 8(%rax)
{disp8} movl $1, 8(%rax)
{disp32} movl $1, 8(%rax)

// Requires no displacement by default
// CHECK: movl $1, (%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x04,0x98,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x44,0x98,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x84,0x98,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%rax,%rbx,4)
{disp8} movl $1, (%rax,%rbx,4)
{disp32} movl $1, (%rax,%rbx,4)

// Requires disp8 by default.
// CHECK: movl $1, 8(%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x44,0x98,0x08,0x01,0x00,0x00,0x00]
// CHECK: movl $1, 8(%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x44,0x98,0x08,0x01,0x00,0x00,0x00]
// CHECK: movl $1, 8(%rax,%rbx,4)
// CHECK: encoding: [0xc7,0x84,0x98,0x08,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, 8(%rax,%rbx,4)
{disp8} movl $1, 8(%rax,%rbx,4)
{disp32} movl $1, 8(%rax,%rbx,4)

// Requires disp8 by default.
// CHECK: movl $1, (%rbp,%rbx,4)
// CHECK: encoding: [0xc7,0x44,0x9d,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rbp,%rbx,4)
// CHECK: encoding: [0xc7,0x44,0x9d,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%rbp,%rbx,4)
// CHECK: encoding: [0xc7,0x84,0x9d,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%rbp,%rbx,4)
{disp8} movl $1, (%rbp,%rbx,4)
{disp32} movl $1, (%rbp,%rbx,4)

// Requires disp8 by default.
// CHECK: movl $1, (%r13,%rbx,4)
// CHECK: encoding: [0x41,0xc7,0x44,0x9d,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%r13,%rbx,4)
// CHECK: encoding: [0x41,0xc7,0x44,0x9d,0x00,0x01,0x00,0x00,0x00]
// CHECK: movl $1, (%r13,%rbx,4)
// CHECK: encoding: [0x41,0xc7,0x84,0x9d,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
movl $1, (%r13,%rbx,4)
{disp8} movl $1, (%r13,%rbx,4)
{disp32} movl $1, (%r13,%rbx,4)

// CHECK: seamcall
// CHECK: encoding: [0x66,0x0f,0x01,0xcf]
seamcall

// CHECK: seamret
// CHECK: encoding: [0x66,0x0f,0x01,0xcd]
seamret

// CHECK: seamops
// CHECK: encoding: [0x66,0x0f,0x01,0xce]
seamops

// CHECK: tdcall
// CHECK: encoding: [0x66,0x0f,0x01,0xcc]
tdcall

// CHECK: hreset
// CHECK: encoding: [0xf3,0x0f,0x3a,0xf0,0xc0,0x01]
hreset $1
