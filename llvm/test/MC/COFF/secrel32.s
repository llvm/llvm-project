// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -S --sr - | FileCheck %s

// check that we produce the correct relocation for .secrel32

Lfoo:
	.secrel32	Lfoo

// CHECK:       Relocations [
// CHECK-NEXT:    0x0 IMAGE_REL_I386_SECREL .text
// CHECK-NEXT:  ]
