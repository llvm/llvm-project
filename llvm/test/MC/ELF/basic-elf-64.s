// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -h -S -r --symbols | FileCheck %s

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
