// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | llvm-readobj -h -s -r -t | FileCheck %s

	.text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# %bb.0:
	subl	$4, %esp
	movl	$.L.str1, (%esp)
	calll	puts
	movl	$.L.str2, (%esp)
	calll	puts
	xorl	%eax, %eax
	addl	$4, %esp
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
// CHECK:   Class: 32-bit
// CHECK:   DataEncoding: LittleEndian
// CHECK:   FileVersion: 1
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: (0)

// CHECK:     Name: .text

// CHECK:     Name: .rel.text

// CHECK: Relocations [
// CHECK:   Section {{.*}} .rel.text {
// CHECK:     0x6  R_386_32   .L.str1
// CHECK:     0xB  R_386_PC32 puts
// CHECK:     0x12 R_386_32   .L.str2
// CHECK:     0x17 R_386_PC32 puts
// CHECK:   }
// CHECK: ]

// CHECK: Symbols [

// CHECK:   Symbol {
// CHECK:     Name: main
// CHECK:     Binding: Global
// CHECK:     Type: Function
// CHECK:   }

// CHECK:   Symbol {
// CHECK:     Name: puts
// CHECK:     Binding: Global
// CHECK:     Type: None
// CHECK:   }
