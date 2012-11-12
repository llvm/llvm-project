// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the common symbols are placed at the end of .bss. In this example
// it causes .bss to have size 9 instead of 8.

	.local	vimvardict
	.comm	vimvardict,1,8
	.bss
        .zero 1
	.align	8

// CHECK:      (('sh_name', 0x00000007) # '.bss'
// CHECK-NEXT:  ('sh_type',
// CHECK-NEXT:  ('sh_flags'
// CHECK-NEXT:  ('sh_addr',
// CHECK-NEXT:  ('sh_offset',
// CHECK-NEXT:  ('sh_size', 0x0000000000000009)
// CHECK-NEXT:  ('sh_link',
// CHECK-NEXT:  ('sh_info',
// CHECK-NEXT:  ('sh_addralign',
// CHECK-NEXT:  ('sh_entsize',
