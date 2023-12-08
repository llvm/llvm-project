// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -h -S -r --symbols - | FileCheck %s

        .text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# %bb.0:
	subq	$8, %rsp
	movl	$.L.str1, %edi
	callq	puts
	movl	$.L.str2, %edi
	callq	puts
	xorl	%eax, %eax
	addq	$8, %rsp
    call foo@GOTPCREL
    ja foo
    jae foo
    jb foo
    jbe foo
    jc foo
    je foo
    jz foo
    jg foo
    jge foo
    jl foo
    jle foo
    jna foo
    jnae foo
    jnb foo
    jnbe foo
    jnc foo
    jne foo
    jng foo
    jnge foo
    jnl foo
    jnle foo
    jno foo
    jnp foo
    jns foo
    jnz foo
    jo foo
    jp foo
    jpe foo
    jpo foo
    js foo
    jz foo
	ret
.Ltmp0:
	.size	main, .Ltmp0-main

	.type	.L.str1,@object         # @.str1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str1:
	.asciz	 "Hello"
	.size	.L.str1, 6

	.type	.L.str2,@object         # @.str2
.L.str2:
	.asciz	 "World!"
	.size	.L.str2, 7

	.section	.note.GNU-stack,"",@progbits

// CHECK: ElfHeader {
// CHECK:   Class: 64-bit
// CHECK:   DataEncoding: LittleEndian
// CHECK:   FileVersion: 1
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: (0)

// CHECK:     Name: .text

// CHECK:     Name: .rela.text

// CHECK:      Relocations [
// CHECK:        Section {{.*}} .rela.text {
// CHECK-NEXT:     0x5  R_X86_64_32   .rodata.str1.1 0x0
// CHECK-NEXT:     0xA  R_X86_64_PLT32 puts           0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xF  R_X86_64_32   .rodata.str1.1 0x6
// CHECK-NEXT:     0x14 R_X86_64_PLT32 puts           0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x1F R_X86_64_GOTPCREL foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x25 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x2B R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x31 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x37 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x3D R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x43 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x49 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x4F R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x55 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x5B R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x61 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x67 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x6D R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x73 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x79 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x7F R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x85 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x8B R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x91 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x97 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x9D R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xA3 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xA9 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xAF R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xB5 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xBB R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xC1 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xC7 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xCD R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xD3 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0xD9 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:   Symbol {
// CHECK:     Binding: Local
// CHECK:     Type: Section

// CHECK:   Symbol {
// CHECK:     Name: main
// CHECK:     Binding: Global
// CHECK:     Type: Function
// CHECK:  }

// CHECK:   Symbol {
// CHECK:     Name: puts
// CHECK:     Binding: Global
// CHECK:     Type: None
// CHECK:  }
