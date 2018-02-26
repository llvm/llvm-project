// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | llvm-readobj -r | FileCheck %s

	.text
	.globl	f
	.type	f,@function
f:                                      # @f
# %bb.0:                                # %entry
	subl	$4, %esp
	calll	42
	incl	%eax
	addl	$4, %esp
	ret
.Ltmp0:
	.size	f, .Ltmp0-f

	.section	.note.GNU-stack,"",@progbits

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) {{[^ ]+}} {
// CHECK-NEXT:     0x4 R_386_PC32 -
// CHECK-NEXT:   }
// CHECK-NEXT: ]
