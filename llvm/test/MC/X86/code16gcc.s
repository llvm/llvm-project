// RUN: llvm-mc -triple i386-unknown-unknown-unknown --show-encoding %s | FileCheck %s 

	.code16gcc
	//CHECK:	.code16
	nop
	//CHECK:	nop                             # encoding: [0x90]
	lodsb
	//CHECK:	lodsb	(%esi), %al             # encoding: [0x67,0xac]
	lodsb (%si), %al
	//CHECK:	lodsb	(%si), %al              # encoding: [0xac]
	lodsb (%esi), %al
	//CHECK:	lodsb	(%esi), %al             # encoding: [0x67,0xac]
	lodsl %gs:(%esi)
	//CHECK:	lodsl	%gs:(%esi), %eax        # encoding: [0x67,0x65,0x66,0xad]
	lods (%esi), %ax
	//CHECK:	lodsw	(%esi), %ax             # encoding: [0x67,0xad]
	stosw
	//CHECK:	stosw	%ax, %es:(%edi)         # encoding: [0x67,0xab]
	stos %eax, (%edi)
	//CHECK:	stosl	%eax, %es:(%edi)        # encoding: [0x67,0x66,0xab]
	stosb %al, %es:(%edi)
	//CHECK:	stosb	%al, %es:(%edi)         # encoding: [0x67,0xaa]
	scas %es:(%edi), %al
	//CHECK:	scasb	%es:(%edi), %al         # encoding: [0x67,0xae]
	scas %es:(%di), %ax
	//CHECK:	scasw	%es:(%di), %ax          # encoding: [0xaf]
	cmpsb
	//CHECK:	cmpsb	%es:(%edi), (%esi)      # encoding: [0x67,0xa6]
	cmpsw (%edi), (%esi)
	//CHECK:	cmpsw	%es:(%edi), (%esi)      # encoding: [0x67,0xa7]
	cmpsl %es:(%edi), %ss:(%esi)
	//CHECK:	cmpsl	%es:(%edi), %ss:(%esi)  # encoding: [0x67,0x36,0x66,0xa7]
	movsb (%esi), (%edi)
	//CHECK:	movsb	(%esi), %es:(%edi)      # encoding: [0x67,0xa4]
	movsl %gs:(%esi), (%edi)
	//CHECK:	movsl	%gs:(%esi), %es:(%edi)  # encoding: [0x67,0x65,0x66,0xa5]
	outsb
	//CHECK:	outsb	(%esi), %dx             # encoding: [0x67,0x6e]
	outsw %fs:(%esi), %dx
	//CHECK:	outsw	%fs:(%esi), %dx         # encoding: [0x67,0x64,0x6f]
	insw %dx, (%di)
	//CHECK:	insw	%dx, %es:(%di)          # encoding: [0x6d]
	call $0x7ace,$0x7ace
	//CHECK:	lcalll	$31438, $31438          # encoding: [0x66,0x9a,0xce,0x7a,0x00,0x00,0xce,0x7a]
	ret
	//CHECK:	retl                            # encoding: [0x66,0xc3]
	pop %ss
	//CHECK:	popl	%ss                     # encoding: [0x66,0x17]
	enter $0x7ace,$0x7f
	//CHECK:	enter	$31438, $127            # encoding: [0xc8,0xce,0x7a,0x7f]
	leave
	//CHECK:	leave                           # encoding: [0xc9]
	push %ss
	//CHECK:	pushl	%ss                     # encoding: [0x66,0x16]
	pop %ss
	//CHECK:	popl	%ss                     # encoding: [0x66,0x17]
	popa
	//CHECK:	popal                           # encoding: [0x66,0x61]
	pushf
	//CHECK:	pushfl                          # encoding: [0x66,0x9c]
	popf
	//CHECK:	popfl                           # encoding: [0x66,0x9d]
	pushw 4
	//CHECK:	pushw	4                       # encoding: [0xff,0x36,0x04,0x00]
	addw $1, (,%eax,4)
	//CHECK:	addw $1, (,%eax,4)              # encoding: [0x67,0x83,0x04,0x85,0x00,0x00,0x00,0x00,0x01]

	

