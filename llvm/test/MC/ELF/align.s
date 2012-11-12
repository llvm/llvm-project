// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the alignment of rodata doesn't force a alignment of the
// previous section (.bss)

	nop
	.section	.rodata,"a",@progbits
	.align	8

// CHECK: # Section 3
// CHECK-NEXT:  (('sh_name', 0x00000007) # '.bss'
// CHECK-NEXT:   ('sh_type', 0x00000008)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000003)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000044)
// CHECK-NEXT:   ('sh_size', 0x0000000000000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 4
// CHECK-NEXT:  (('sh_name', 0x00000026) # '.rodata'
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000002)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000048)
// CHECK-NEXT:   ('sh_size', 0x0000000000000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000008)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
