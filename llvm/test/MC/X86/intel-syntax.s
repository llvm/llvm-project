// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

_test:
	xor	EAX, EAX
	ret

.set  number, 8
.global _foo

.text
  .global main
main:

// CHECK: leaq    _foo(%rbx,%rax,8), %rdx
  lea RDX, [8 * RAX + RBX      + _foo]
// CHECK: leaq _foo(%rbx,%rax,8), %rdx
  lea RDX, [_foo + 8 * RAX + RBX]
// CHECK: leaq 8(%rcx,%rax,8), %rdx
  lea RDX, [8 + RAX * 8 + RCX]
// CHECK: leaq 8(%rcx,%rax,8), %rdx
  lea RDX, [number + 8 * RAX + RCX]
// CHECK: leaq _foo(,%rax,8), %rdx
  lea RDX, [_foo + RAX * 8]
// CHECK:  leaq _foo(%rbx,%rax,8), %rdx
  lea RDX, [_foo + RAX * 8 + RBX]
// CHECK: leaq -8(%rax), %rdx
  lea RDX, [RAX - number]
// CHECK: leaq -8(%rax), %rdx
  lea RDX, [RAX - 8]
// CHECK: leaq    _foo(%rax), %rdx
  lea RDX, [RAX + _foo]
// CHECK: leaq    8(%rax), %rdx
  lea RDX, [RAX + number]
// CHECK: leaq    8(%rax), %rdx
  lea RDX, [RAX + 8]
// CHECK: leaq    _foo(%rbx,%rax,8), %rdx
  lea RDX, [RAX * number + RBX + _foo]
// CHECK: leaq    _foo(%rbx,%rax,8), %rdx
  lea RDX, [_foo + RAX * number + RBX]
// CHECK: leaq    8(%rcx,%rax,8), %rdx
  lea RDX, [number + RAX * number + RCX]
// CHECK: leaq    _foo(,%rax,8), %rdx
  lea RDX, [_foo + RAX * number]
// CHECK: leaq    _foo(%rbx,%rax,8), %rdx
  lea RDX, [number * RAX + RBX + _foo]
// CHECK: leaq    _foo(%rbx,%rax,8), %rdx
  lea RDX, [_foo + number * RAX + RBX]
// CHECK: leaq    8(%rcx,%rax,8), %rdx
  lea RDX, [8 + number * RAX + RCX]
// CHECK: leaq    _foo(%rax), %rdx
  lea RDX, [_foo + RAX]
// CHECK: leaq    8(%rax), %rdx
  lea RDX, [number + RAX]
// CHECK: leaq    8(%rax), %rdx
  lea RDX, [8 + RAX]

// CHECK: lcalll *(%rax)
  call FWORD ptr [rax]
// CHECK: lcalll *(%rax)
  lcall [rax]
// CHECK: ljmpl *(%rax)
  jmp FWORD ptr [rax]
// CHECK: ljmpq *(%rax)
  ljmp [rax]
// CHECK: jmp _foo
  jmp short _foo
// CHECK: jb _foo
  jc short _foo
// CHECK: jae _foo
  jnc short _foo
// CHECK: jecxz _foo
  jecxz short _foo
// CHECK: jp _foo
  jpe short _foo

// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257
// CHECK:	movl	$258, 4(%rsp)
	mov	DWORD PTR [RSP + 4], 258
// CHECK:	movq	$123, -16(%rsp)
	mov	QWORD PTR [RSP - 16], 123
// CHECK:	movb	$97, -17(%rsp)
	mov	BYTE PTR [RSP - 17], 97
// CHECK:	movl	-4(%rsp), %eax
	mov	EAX, DWORD PTR [RSP - 4]
// CHECK:	movq    (%rsp), %rax
	mov     RAX, QWORD PTR [RSP]
// CHECK: movabsq $4294967289, %rax
	mov     RAX, 4294967289
// CHECK:	movl	$-4, -4(%rsp)
	mov	DWORD PTR [RSP - 4], -4
// CHECK:	movq	0, %rcx
	mov	RCX, QWORD PTR [0]
// CHECK:	movl	-24(%rsp,%rax,4), %eax	
	mov	EAX, DWORD PTR [RSP + 4*RAX - 24]
// CHECK:	movb	%dil, (%rdx,%rcx)
	mov	BYTE PTR [RDX + RCX], DIL
// CHECK:	movzwl	2(%rcx), %edi
	movzx	EDI, WORD PTR [RCX + 2]
// CHECK:	callq	_test
	call	_test
// CHECK:	andw	$12,	%ax
	and	ax, 12
// CHECK:	andw	$-12,	%ax
	and	ax, -12
// CHECK:	andw	$257,	%ax
	and	ax, 257
// CHECK:	andw	$-257,	%ax
	and	ax, -257
// CHECK:	andl	$12,	%eax
	and	eax, 12
// CHECK:	andl	$-12,	%eax
	and	eax, -12
// CHECK:	andl	$257,	%eax
	and	eax, 257
// CHECK:	andl	$-257,	%eax
	and	eax, -257
// CHECK:	andq	$12,	%rax
	and	rax, 12
// CHECK:	andq	$-12,	%rax
	and	rax, -12
// CHECK:	andq	$257,	%rax
	and	rax, 257
// CHECK:	andq	$-257,	%rax
	and	rax, -257
// CHECK:	fld	%st(0)
	fld	ST(0)
// CHECK:	movl	%fs:(%rdi), %eax
    mov EAX, DWORD PTR FS:[RDI]
// CHECK: leal (,%rdi,4), %r8d
    lea R8D, DWORD PTR [4*RDI]
// CHECK: movl _fnan(,%ecx,4), %ecx
    mov ECX, DWORD PTR [4*ECX + _fnan]
// CHECK: movq %fs:320, %rax
    mov RAX, QWORD PTR FS:[320]
// CHECK: movq %fs:320, %rax
    mov RAX, QWORD PTR FS:320
// CHECK: movq %rax, %fs:320
    mov QWORD PTR FS:320, RAX
// CHECK: movq %rax, %fs:20(%rbx)
    mov QWORD PTR FS:20[rbx], RAX
// CHECK: vshufpd $1, %xmm2, %xmm1, %xmm0
    vshufpd XMM0, XMM1, XMM2, 1
// CHECK: vpgatherdd %xmm8, (%r15,%xmm9,2), %xmm1
    vpgatherdd XMM10, XMMWORD PTR [R15 + 2*XMM9], XMM8
// CHECK: movsd -8, %xmm5
    movsd   XMM5, QWORD PTR [-8]
// CHECK: movsl (%rsi), %es:(%rdi)
    movsd
// CHECK: movl %ecx, (%eax)
    mov [eax], ecx
// CHECK: movl %ecx, (,%ebx,4)
    mov [4*ebx], ecx
 // CHECK:   movl %ecx, (,%ebx,4)
    mov [ebx*4], ecx
// CHECK: movl %ecx, 1024
    mov [1024], ecx
// CHECK: movl %ecx, 4132
    mov [0x1024], ecx
// CHECK: movl %ecx, 32        
    mov [16 + 16], ecx
// CHECK: movl %ecx, 0
    mov [16 - 16], ecx        
// CHECK: movl %ecx, 32        
    mov [16][16], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax + 4*ebx], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax + ebx*4], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [4*ebx + eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [ebx*4 + eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax][4*ebx], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [eax][ebx*4], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [4*ebx][eax], ecx
// CHECK: movl %ecx, (%eax,%ebx,4)
    mov [ebx*4][eax], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [eax + 12], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [12 + eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax + 16 + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + eax + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + 16 + eax], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [eax][12], ecx
// CHECK: movl %ecx, 12(%eax)
    mov [12][eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16 + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax + 16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][eax + 16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + eax][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][16 + eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16 + 16][eax], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [eax][16][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][eax][16], ecx
// CHECK: movl %ecx, 32(%eax)
    mov [16][16][eax], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [4*ebx + 16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [4*ebx][16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [ebx*4][16], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16 + ebx*4], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16][4*ebx], ecx
// CHECK: movl %ecx, 16(,%ebx,4)
    mov [16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 4*ebx + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + 16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][4*ebx + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16 + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + 4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 4*ebx][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx + 16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][4*ebx][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [4*ebx][16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][4*ebx], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16 + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + 16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][ebx*4 + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16 + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][eax + 16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][16 + eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax + ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + ebx*4][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax + 16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4 + 16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16 + eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][ebx*4][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [eax][16][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][eax][16], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [ebx*4][16][eax], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][ebx*4], ecx
// CHECK: movl %ecx, 16(%eax,%ebx,4)
    mov [16][eax][ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 16], ecx

// CHECK: prefetchnta 12800(%esi)
    prefetchnta [esi + (200*64)]
// CHECK: prefetchnta 32(%esi)
    prefetchnta [esi + (64/2)]
// CHECK: prefetchnta 128(%esi)
    prefetchnta [esi + (64/2*4)]
// CHECK: prefetchnta 8(%esi)
    prefetchnta [esi + (64/(2*4))]
// CHECK: prefetchnta 48(%esi)
    prefetchnta [esi + (64/(2*4)+40)]

// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][4*ebx - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + 4*ebx - 2*8], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [12 + eax + (4*ebx) - 2*14], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2*2*2*2], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - (2*8)], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax][ebx*4 - 2 * 8 + 4 - 4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ebx*4 - 2 * 8 + 4 - 4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ebx*4 - 2 * ((8 + 4) - 4)], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [-2 * ((8 + 4) - 4) + eax + ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [((-2) * ((8 + 4) - 4)) + eax + ebx*4], ecx
// CHECK: movl %ecx, -16(%eax,%ebx,4)
    mov [eax + ((-2) * ((8 + 4) - 4)) + ebx*4], ecx
// CHECK: movl %ecx, 96(%eax,%ebx,4)
    mov [eax + ((-2) * ((8 + 4) * -4)) + ebx*4], ecx
// CHECK: movl %ecx, -8(%eax,%ebx,4)
    mov [eax][-8][ebx*4], ecx
// CHECK: movl %ecx, -2(%eax,%ebx,4)
    mov [eax][16/-8][ebx*4], ecx
// CHECK: movl %ecx, -2(%eax,%ebx,4)
    mov [eax][(16)/-8][ebx*4], ecx

// CHECK: setb %al
    setc al
// CHECK: sete %al
    setz al
// CHECK: setbe %al
    setna al
// CHECK: setae %al
    setnb al
// CHECK: setae %al
    setnc al
// CHECK: setle %al
    setng al
// CHECK: setge %al
    setnl al
// CHECK: setne %al
    setnz al
// CHECK: setp %al
    setpe al
// CHECK: setnp %al
    setpo al
// CHECK: setb %al
    setnae al
// CHECK: seta %al
    setnbe al
// CHECK: setl %al
    setnge al
// CHECK: setg %al
    setnle al
// CHECK: jne _foo
    jnz _foo
// CHECK: outb %al, $4
    out 4, al
    ret

// CHECK: cmovbl %ebx, %eax
    cmovc eax, ebx
// CHECK: cmovel %ebx, %eax
    cmovz eax, ebx
// CHECK: cmovbel %ebx, %eax
    cmovna eax, ebx
// CHECK: cmovael %ebx, %eax
    cmovnb eax, ebx
// CHECK: cmovael %ebx, %eax
    cmovnc eax, ebx
// CHECK: cmovlel %ebx, %eax
    cmovng eax, ebx
// CHECK: cmovgel %ebx, %eax
    cmovnl eax, ebx
// CHECK: cmovnel %ebx, %eax
    cmovnz eax, ebx
// CHECK: cmovpl %ebx, %eax
    cmovpe eax, ebx
// CHECK: cmovnpl %ebx, %eax
    cmovpo eax, ebx
// CHECK: cmovbl %ebx, %eax
    cmovnae eax, ebx
// CHECK: cmoval %ebx, %eax
    cmovnbe eax, ebx
// CHECK: cmovll %ebx, %eax
    cmovnge eax, ebx
// CHECK: cmovgl %ebx, %eax
    cmovnle eax, ebx

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

shld  DX, BX
shld  DX, BX, CL
shld  DX, BX, 1
shld  [RAX], BX
shld  [RAX], BX, CL
shrd  DX, BX
shrd  DX, BX, CL
shrd  DX, BX, 1
shrd  [RAX], BX
shrd  [RAX], BX, CL

// CHECK: btl $1, (%eax)
// CHECK: btsl $1, (%eax)
// CHECK: btrl $1, (%eax)
// CHECK: btcl $1, (%eax)
    bt DWORD PTR [EAX], 1
    bt DWORD PTR [EAX], 1
    bts DWORD PTR [EAX], 1
    btr DWORD PTR [EAX], 1
    btc DWORD PTR [EAX], 1

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
    div AL, BL
    div AX, BX
    div EAX, ECX
    div EAX, [ECX*8+EBX+0xdeadbeef]
    div EAX, [0x45]
    div EAX, [0x7eed]
    div EAX, [0xbabecafe]
    div EAX, [0x12345678]
    idiv AL, BL
    idiv AX, BX
    idiv EAX, ECX
    idiv EAX, [ECX*8+EBX+0xdeadbeef]
    idiv EAX, [0x45]
    idiv EAX, [0x7eed]
    idiv EAX, [0xbabecafe]
    idiv EAX, [0x12345678]


// CHECK: inb %dx, %al
// CHECK: inw %dx, %ax
// CHECK: inl %dx, %eax
// CHECK: outb %al, %dx
// CHECK: outw %ax, %dx
// CHECK: outl %eax, %dx
    inb DX
    inw DX
    inl DX
    outb DX
    outw DX
    outl DX

// CHECK: xchgq %rcx, %rax
// CHECK: xchgq %rcx, %rax
// CHECK: xchgl %ecx, %eax
// CHECK: xchgl %ecx, %eax
// CHECK: xchgw %cx, %ax
// CHECK: xchgw %cx, %ax
xchg RAX, RCX
xchg RCX, RAX
xchg EAX, ECX
xchg ECX, EAX
xchg AX, CX
xchg CX, AX

// CHECK: xchgq %rax, (%ecx)
// CHECK: xchgq %rax, (%ecx)
// CHECK: xchgl %eax, (%ecx)
// CHECK: xchgl %eax, (%ecx)
// CHECK: xchgw %ax, (%ecx)
// CHECK: xchgw %ax, (%ecx)
xchg RAX, [ECX]
xchg [ECX], RAX
xchg EAX, [ECX]
xchg [ECX], EAX
xchg AX, [ECX]
xchg [ECX], AX

// CHECK: testq %rax, (%ecx)
// CHECK: testq %rax, (%ecx)
// CHECK: testl %eax, (%ecx)
// CHECK: testl %eax, (%ecx)
// CHECK: testw %ax, (%ecx)
// CHECK: testw %ax, (%ecx)
// CHECK: testb %al, (%ecx)
// CHECK: testb %al, (%ecx)
test RAX, [ECX]
test [ECX], RAX
test EAX, [ECX]
test [ECX], EAX
test AX, [ECX]
test [ECX], AX
test AL, [ECX]
test [ECX], AL

// CHECK: fnstsw %ax
// CHECK: fnstsw %ax
// CHECK: fnstsw (%eax)
fnstsw
fnstsw AX
fnstsw WORD PTR [EAX]

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
faddp ST(1), ST(0)
fmulp ST(1), ST(0)
fsubp ST(1), ST(0)
fsubrp ST(1), ST(0)
fdivp ST(1), ST(0)
fdivrp ST(1), ST(0)

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
faddp ST(0), ST(1)
fmulp ST(0), ST(1)
fsubp ST(0), ST(1)
fsubrp ST(0), ST(1)
fdivp ST(0), ST(1)
fdivrp ST(0), ST(1)

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
faddp ST(1)
fmulp ST(1)
fsubp ST(1)
fsubrp ST(1)
fdivp ST(1)
fdivrp ST(1)


// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
fadd 
fmul
fsub
fsubr
fdiv
fdivr

// CHECK: faddp %st, %st(1)
// CHECK: fmulp %st, %st(1)
// CHECK: fsubrp %st, %st(1)
// CHECK: fsubp %st, %st(1)
// CHECK: fdivrp %st, %st(1)
// CHECK: fdivp %st, %st(1)
faddp
fmulp
fsubp
fsubrp
fdivp
fdivrp

// CHECK: fadd %st(1), %st
// CHECK: fmul %st(1), %st
// CHECK: fsub %st(1), %st
// CHECK: fsubr %st(1), %st
// CHECK: fdiv %st(1), %st
// CHECK: fdivr %st(1), %st
fadd ST(0), ST(1)
fmul ST(0), ST(1)
fsub ST(0), ST(1)
fsubr ST(0), ST(1)
fdiv ST(0), ST(1)
fdivr ST(0), ST(1)

// CHECK: fadd %st, %st(1)
// CHECK: fmul %st, %st(1)
// CHECK: fsubr %st, %st(1)
// CHECK: fsub %st, %st(1)
// CHECK: fdivr %st, %st(1)
// CHECK: fdiv %st, %st(1)
fadd ST(1), ST(0)
fmul ST(1), ST(0)
fsub ST(1), ST(0)
fsubr ST(1), ST(0)
fdiv ST(1), ST(0)
fdivr ST(1), ST(0)

// CHECK: fadd %st(1), %st
// CHECK: fmul %st(1), %st
// CHECK: fsub %st(1), %st
// CHECK: fsubr %st(1), %st
// CHECK: fdiv %st(1), %st
// CHECK: fdivr %st(1), %st
fadd ST(1)
fmul ST(1)
fsub ST(1)
fsubr ST(1)
fdiv ST(1)
fdivr ST(1)


// CHECK: fxsave64 (%rax)
// CHECK: fxrstor64 (%rax)
fxsave64 [rax]
fxrstor64 [rax]

.bss
.globl _g0
.text

// CHECK: movq _g0, %rbx
// CHECK: movq _g0+8, %rcx
// CHECK: movq _g0+18(%rbp), %rax
// CHECK: movq _g0(,%rsi,4), %rax
mov rbx, qword ptr [_g0]
mov rcx, qword ptr [_g0 + 8]
mov rax, QWORD PTR _g0[rbp + 1 + (2 * 5) - 3 + 1<<1]
mov rax, QWORD PTR _g0[rsi*4]

"?half@?0??bar@@YAXXZ@4NA":
	.quad   4602678819172646912

fadd   dword ptr "?half@?0??bar@@YAXXZ@4NA"
fadd   dword ptr "?half@?0??bar@@YAXXZ@4NA"@IMGREL
// CHECK: fadds   "?half@?0??bar@@YAXXZ@4NA"
// CHECK: fadds   "?half@?0??bar@@YAXXZ@4NA"@IMGREL

inc qword ptr [rax]
inc long ptr [rax]
inc dword ptr [rax]
inc word ptr [rax]
inc byte ptr [rax]
// CHECK: incq (%rax)
// CHECK: incl (%rax)
// CHECK: incl (%rax)
// CHECK: incw (%rax)
// CHECK: incb (%rax)

dec qword ptr [rax]
dec dword ptr [rax]
dec word ptr [rax]
dec byte ptr [rax]
// CHECK: decq (%rax)
// CHECK: decl (%rax)
// CHECK: decw (%rax)
// CHECK: decb (%rax)

add qword ptr [rax], 1
add dword ptr [rax], 1
add word ptr [rax], 1
add byte ptr [rax], 1
// CHECK: addq $1, (%rax)
// CHECK: addl $1, (%rax)
// CHECK: addw $1, (%rax)
// CHECK: addb $1, (%rax)

fstp tbyte ptr [rax]
fstp xword ptr [rax]
fstp qword ptr [rax]
fstp dword ptr [rax]
// CHECK: fstpt (%rax)
// CHECK: fstpt (%rax)
// CHECK: fstpl (%rax)
// CHECK: fstps (%rax)

fxsave [eax]
fsave [eax]
fxrstor [eax]
frstor [eax]
// CHECK: fxsave (%eax)
// CHECK: wait
// CHECK: fnsave (%eax)
// CHECK: fxrstor (%eax)
// CHECK: frstor (%eax)

// FIXME: Should we accept this?  Masm accepts it, but gas does not.
fxsave dword ptr [eax]
fsave dword ptr [eax]
fxrstor dword ptr [eax]
frstor dword ptr [eax]
// CHECK: fxsave (%eax)
// CHECK: wait
// CHECK: fnsave (%eax)
// CHECK: fxrstor (%eax)
// CHECK: frstor (%eax)

// CHECK: cmpnless %xmm1, %xmm0
cmpnless xmm0, xmm1

insb
insw
insd
// CHECK: insb %dx, %es:(%rdi)
// CHECK: insw %dx, %es:(%rdi)
// CHECK: insl %dx, %es:(%rdi)

outsb
outsw
outsd
// CHECK: outsb (%rsi), %dx
// CHECK: outsw (%rsi), %dx
// CHECK: outsl (%rsi), %dx

imul bx, 123
imul ebx, 123
imul rbx, 123
// CHECK: imulw $123, %bx
// CHECK: imull $123, %ebx
// CHECK: imulq $123, %rbx

repe cmpsb
repz cmpsb
repne cmpsb
repnz cmpsb
// CHECK: rep
// CHECK: cmpsb	%es:(%rdi), (%rsi)
// CHECK: rep
// CHECK: cmpsb	%es:(%rdi), (%rsi)
// CHECK: repne
// CHECK: cmpsb	%es:(%rdi), (%rsi)
// CHECK: repne
// CHECK: cmpsb	%es:(%rdi), (%rsi)

sal eax, 123
// CHECK: shll	$123, %eax

psignw    mm0, MMWORD PTR t2
// CHECK: psignw t2, %mm0

comisd xmm0, QWORD PTR [eax]
comiss xmm0, DWORD PTR [eax]
vcomisd xmm0, QWORD PTR [eax]
vcomiss xmm0, DWORD PTR [eax]

// CHECK: comisd (%eax), %xmm0
// CHECK: comiss (%eax), %xmm0
// CHECK: vcomisd (%eax), %xmm0
// CHECK: vcomiss (%eax), %xmm0

fbld tbyte ptr [eax]
fbstp tbyte ptr [eax]
// CHECK: fbld (%eax)
// CHECK: fbstp (%eax)

fld float ptr [rax]
fld double ptr [rax]
// CHECK: flds (%rax)
// CHECK: fldl (%rax)

fcomip st, st(2)
fucomip st, st(2)
// CHECK: fcompi  %st(2)
// CHECK: fucompi  %st(2)

loopz _foo
loopnz _foo
// CHECK: loope _foo
// CHECK: loopne _foo

sidt fword ptr [eax]
// CHECK: sidtq (%eax)

ins byte ptr [eax], dx
// CHECK: insb %dx, %es:(%edi)
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)DI will be used for the location
// CHECK-STDERR-NEXT: ins byte ptr [eax], dx
outs dx, word ptr [eax]
// CHECK: outsw (%esi), %dx
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)SI will be used for the location
// CHECK-STDERR-NEXT: outs dx, word ptr [eax]
lods dword ptr [eax]
// CHECK: lodsl (%esi), %eax
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)SI will be used for the location
// CHECK-STDERR-NEXT: lods dword ptr [eax]
stos qword ptr [eax]
// CHECK: stosq %rax, %es:(%edi)
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)DI will be used for the location
// CHECK-STDERR-NEXT: stos qword ptr [eax]
scas byte ptr [eax]
// CHECK: scasb %es:(%edi), %al
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)DI will be used for the location
// CHECK-STDERR-NEXT: scas byte ptr [eax]
cmps word ptr [eax], word ptr [ebx]
// CHECK: cmpsw %es:(%edi), (%esi)
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)SI will be used for the location
// CHECK-STDERR-NEXT: cmps word ptr [eax], word ptr [ebx]
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)DI will be used for the location
// CHECK-STDERR-NEXT: cmps word ptr [eax], word ptr [ebx]
movs dword ptr [eax], dword ptr [ebx]
// CHECK: movsl (%esi), %es:(%edi)
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)DI will be used for the location
// CHECK-STDERR-NEXT: movs dword ptr [eax], dword ptr [ebx]
// CHECK-STDERR: memory operand is only for determining the size, ES:(R|E)SI will be used for the location
// CHECK-STDERR-NEXT: movs dword ptr [eax], dword ptr [ebx]

movsd  qword ptr [rax], xmm0
// CHECK: movsd %xmm0, (%rax)
// CHECK-STDERR-NOT: movsd qword ptr [rax], xmm0

xlat byte ptr [eax]
// CHECK: xlatb
// CHECK-STDERR: memory operand is only for determining the size, (R|E)BX will be used for the location

// CHECK:   punpcklbw
punpcklbw mm0, dword ptr [rsp]
// CHECK:   punpcklwd
punpcklwd mm0, dword ptr [rsp]
// CHECK:   punpckldq
punpckldq mm0, dword ptr [rsp]

// CHECK: lslq (%eax), %rbx
lsl rbx, word ptr [eax]

// CHECK: lsll (%eax), %ebx
lsl ebx, word ptr [eax]

// CHECK: lslw (%eax), %bx
lsl bx, word ptr [eax]

// CHECK: sysexitl
sysexit
// CHECK: sysexitq
sysexitq
// CHECK: sysretl
sysret
// CHECK: sysretq
sysretq

// CHECK: leaq (%rsp,%rax), %rax
lea rax, [rax+rsp]
// CHECK: leaq (%rsp,%rax), %rax
lea rax, [rsp+rax]
// CHECK: leal (%esp,%eax), %eax
lea eax, [eax+esp]
// CHECK: leal (%esp,%eax), %eax
lea eax, [esp+eax]

// CHECK: vpgatherdq      %ymm2, (%rdi,%xmm1), %ymm0
vpgatherdq ymm0, [rdi+xmm1], ymm2
// CHECK: vpgatherdq      %ymm2, (%rdi,%xmm1), %ymm0
vpgatherdq ymm0, [xmm1+rdi], ymm2
