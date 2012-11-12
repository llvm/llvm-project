// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

	.text
	.globl	f
	.type	f,@function
f:                                      # @f
# BB#0:                                 # %entry
	subl	$4, %esp
	calll	42
	incl	%eax
	addl	$4, %esp
	ret
.Ltmp0:
	.size	f, .Ltmp0-f

	.section	.note.GNU-stack,"",@progbits

// CHECK:      ('_relocations', [
// CHECK-NEXT:  # Relocation 0
// CHECK-NEXT:  (('r_offset', 0x00000004)
// CHECK-NEXT:   ('r_sym', 0x000000)
// CHECK-NEXT:   ('r_type', 0x02)
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
