// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

// Test that .debug_line is populated.

// CHECK:     (('sh_name', 0x00000011) # '.debug_line'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x0000000000000000)
// CHECK-NEXT: ('sh_addr', 0x0000000000000000)
// CHECK-NEXT: ('sh_offset', 0x0000000000000044)
// CHECK-NEXT: ('sh_size', 0x0000000000000037)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT: ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ('_section_data', '33000000 02001c00 00000101 fb0e0d00 01010101 00000001 00000100 666f6f2e 63000000 00000009 02000000 00000000 00150204 000101')

	.section	.debug_line,"",@progbits
	.text

	.file 1 "foo.c"
	.loc 1 4 0
	subq	$8, %rsp
