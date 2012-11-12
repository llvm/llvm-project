// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the bss section is correctly aligned

	.local	foo
	.comm	foo,2048,16

// CHECK:        ('sh_name', 0x00000007) # '.bss'
// CHECK-NEXT:   ('sh_type', 0x00000008)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000003)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:   ('sh_size', 0x0000000000000800)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000010)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
