// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

        .text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# BB#0:
	subq	$8, %rsp
	movl	$.L.str1, %edi
	callq	puts
	movl	$.L.str2, %edi
	callq	puts
	xorl	%eax, %eax
	addq	$8, %rsp
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

// CHECK: ('e_indent[EI_CLASS]', 0x02)
// CHECK: ('e_indent[EI_DATA]', 0x01)
// CHECK: ('e_indent[EI_VERSION]', 0x01)
// CHECK: ('_sections', [
// CHECK:   # Section 0
// CHECK:   (('sh_name', 0x00000000) # ''

// CHECK:   # '.text'

// CHECK:   # '.rela.text'

// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('r_offset', 0x0000000000000005)
// CHECK:      ('r_type', 0x0000000a)
// CHECK:      ('r_addend', 0x0000000000000000)
// CHECK:     ),
// CHECK:     # Relocation 1
// CHECK:     (('r_offset', 0x000000000000000a)
// CHECK:      ('r_type', 0x00000002)
// CHECK:      ('r_addend', 0xfffffffffffffffc)
// CHECK:     ),
// CHECK:     # Relocation 2
// CHECK:     (('r_offset', 0x000000000000000f)
// CHECK:      ('r_type', 0x0000000a)
// CHECK:      ('r_addend', 0x0000000000000006)
// CHECK:     ),
// CHECK:     # Relocation 3
// CHECK:     (('r_offset', 0x0000000000000014)
// CHECK:      ('r_type', 0x00000002)
// CHECK:      ('r_addend', 0xfffffffffffffffc)
// CHECK:     ),
// CHECK:   ])

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK:   # 'main'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x2)

// CHECK:   # 'puts'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x0)
