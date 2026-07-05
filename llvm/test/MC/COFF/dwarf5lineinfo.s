// RUN: llvm-mc -filetype=obj -triple x86_64-pc-windows-gnu %s -o - | llvm-readobj -r  - | FileCheck %s

// CHECK: Relocations [
// CHECK:  Section (4) .debug_line {
// CHECK:    0x22 IMAGE_REL_AMD64_SECREL .debug_line_str (8)
// CHECK:    0x2C IMAGE_REL_AMD64_SECREL .debug_line_str (8)
// CHECK:    0x36 IMAGE_REL_AMD64_ADDR64 .text (0)
// CHECK:  }

main:
	.file	0 "/" "test.c"
	.loc	0 1 0
	retq
