// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the alignment does contribute to the size of the section.

	.zero 4
	.align	8

// CHECK:      (('sh_name', 0x00000001) # '.text'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000006)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:  ('sh_size', 0x0000000000000008)
