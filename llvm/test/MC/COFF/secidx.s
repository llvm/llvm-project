// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -S --sr | FileCheck %s

// check that we produce the correct relocation for .secidx

Lfoo:
	.secidx	Lfoo
	.short  0
	.secidx	Lbar
	.short  0

.section spam
Lbar:
	ret

// CHECK:       Relocations [
// CHECK-NEXT:    0x0 IMAGE_REL_I386_SECTION .text
// CHECK-NEXT:    0x4 IMAGE_REL_I386_SECTION spam
// CHECK-NEXT:  ]
