// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that mergeable constants have sh_entsize set.

// 1 byte strings
    .section	.rodata.str1.1,"aMS",@progbits,1

    .type	.L.str1,@object         # @.str1
.L.str1:
	.asciz	 "tring"
	.size	.L.str1, 6

	.type	.L.str2,@object         # @.str2
.L.str2:
	.asciz	 "String"
	.size	.L.str2, 7

// 2 byte strings
    .section	.rodata.str2.1,"aMS",@progbits,2
	.type	.L.str3,@object         # @.str3
.L.str3:
	.asciz	 "L\000o\000n\000g\000"
	.size	.L.str3, 9

	.type	.L.str4,@object         # @.str4
.L.str4:
	.asciz	 "o\000n\000g\000"
	.size	.L.str4, 7

 // 8 byte constants
    .section	.rodata.cst8,"aM",@progbits,8
    .quad 42
    .quad 42

// CHECK: # Section 4
// CHECK-NEXT:   ('sh_name', 0x00000048) # '.rodata.str1.1'
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000032)
// CHECK-NEXT:   ('sh_addr',
// CHECK-NEXT:   ('sh_offset',
// CHECK-NEXT:   ('sh_size', 0x000000000000000d)
// CHECK-NEXT:   ('sh_link',
// CHECK-NEXT:   ('sh_info',
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000001)

// CHECK: # Section 5
// CHECK-NEXT:   ('sh_name', 0x00000039) # '.rodata.str2.1'
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000032)
// CHECK-NEXT:   ('sh_addr',
// CHECK-NEXT:   ('sh_offset',
// CHECK-NEXT:   ('sh_size', 0x0000000000000010)
// CHECK-NEXT:   ('sh_link',
// CHECK-NEXT:   ('sh_info',
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000002)

// CHECK: # Section 6
// CHECK-NEXT:   ('sh_name', 0x0000002c) # '.rodata.cst8
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000012)
// CHECK-NEXT:   ('sh_addr',
// CHECK-NEXT:   ('sh_offset',
// CHECK-NEXT:   ('sh_size', 0x0000000000000010)
// CHECK-NEXT:   ('sh_link',
// CHECK-NEXT:   ('sh_info',
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000008)
