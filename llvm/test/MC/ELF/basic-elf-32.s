// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

	.text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# BB#0:
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

// CHECK: ('e_indent[EI_CLASS]', 0x01)
// CHECK: ('e_indent[EI_DATA]', 0x01)
// CHECK: ('e_indent[EI_VERSION]', 0x01)
// CHECK: ('_sections', [
// CHECK:   # Section 0
// CHECK:   (('sh_name', 0x00000000) # ''

// CHECK:   # '.text'

// CHECK:   # '.rel.text'

// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('r_offset', 0x00000006)
// CHECK:      ('r_type', 0x01)
// CHECK:     ),
// CHECK:     # Relocation 1
// CHECK:     (('r_offset', 0x0000000b)
// CHECK:      ('r_type', 0x02)
// CHECK:     ),
// CHECK:     # Relocation 2
// CHECK:     (('r_offset', 0x00000012)
// CHECK:      ('r_type', 0x01)
// CHECK:     ),
// CHECK:     # Relocation 3
// CHECK:     (('r_offset', 0x00000017)
// CHECK:      ('r_type', 0x02)
// CHECK:     ),
// CHECK:   ])

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK: ('st_bind', 0x0)
// CHECK: ('st_type', 0x3)

// CHECK:   # 'main'
// CHECK:   ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x2)

// CHECK:   # 'puts'
// CHECK:   ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x0)
