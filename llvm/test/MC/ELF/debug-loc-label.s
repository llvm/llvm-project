// Verify that the .loc_label instruction resets the line sequence and generates
// the requested label at the correct position in the line stream

// RUN: llvm-mc -filetype obj -triple x86_64-linux-elf %s -o %t.o
// RUN: llvm-dwarfdump -v --debug-line %t.o | FileCheck %s --check-prefix=CHECK-LINE-TABLE
// RUN: llvm-objdump -s -j .offset_02 -j .offset_03 -j .offset_05 %t.o | FileCheck %s --check-prefix=CHECK-SECTIONS



# CHECK-LINE-TABLE:                  Address            Line   Column File   ISA Discriminator OpIndex Flags
# CHECK-LINE-TABLE-NEXT:             ------------------ ------ ------ ------ --- ------------- ------- -------------
# CHECK-LINE-TABLE-NEXT: 0x00000028: 05 DW_LNS_set_column (1)
# CHECK-LINE-TABLE-NEXT: 0x0000002a: 00 DW_LNE_set_address (0x0000000000000000)
# CHECK-LINE-TABLE-NEXT: 0x00000035: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000000      1      1      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000036: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000000      1      1      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x00000039: 05 DW_LNS_set_column (2)
# CHECK-LINE-TABLE-NEXT: 0x0000003b: 00 DW_LNE_set_address (0x0000000000000008)
# CHECK-LINE-TABLE-NEXT: 0x00000046: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000008      1      2      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000047: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000008      1      2      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000004a: 05 DW_LNS_set_column (3)
# CHECK-LINE-TABLE-NEXT: 0x0000004c: 00 DW_LNE_set_address (0x0000000000000010)
# CHECK-LINE-TABLE-NEXT: 0x00000057: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000010      1      3      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000058: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000010      1      3      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000005b: 05 DW_LNS_set_column (4)
# CHECK-LINE-TABLE-NEXT: 0x0000005d: 00 DW_LNE_set_address (0x0000000000000018)
# CHECK-LINE-TABLE-NEXT: 0x00000068: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      4      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000069: 05 DW_LNS_set_column (5)
# CHECK-LINE-TABLE-NEXT: 0x0000006b: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      5      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x0000006c: 02 DW_LNS_advance_pc (addr += 9, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x0000006e: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000021      1      5      1   0             0       0  is_stmt end_sequence

# CHECK-SECTIONS: Contents of section .offset_02:
# CHECK-SECTIONS-NEXT: 0000 39000000

# CHECK-SECTIONS: Contents of section .offset_03:
# CHECK-SECTIONS-NEXT: 0000 4a000000

# CHECK-SECTIONS: Contents of section .offset_05:
# CHECK-SECTIONS-NEXT: 0000 5b000000
	.text
	.file	"test.c"
	.globl	foo
	.align	16, 0x90
	.type	foo,@function
foo:
.Lfunc_begin0:
	.file	1 "test.c"
	.cfi_startproc
	.loc	1 1 1
	mov     %rax, 0x01
	.loc_label my_label_02
	.loc	1 1 2
	mov     %rax, 0x02
	.loc	1 1 3
	.loc_label my_label_03
	.loc_label my_label_03.1
	mov     %rax, 0x03
	.loc	1 1 4
	.loc_label my_label_04
	.loc	1 1 5
	mov     %rax, 0x04
	ret
	.cfi_endproc

	.section	.debug_line,"",@progbits
.Lline_table_start0:

	.section	.offset_02,"",@progbits
	.quad	my_label_02-.Lline_table_start0

	.section	.offset_03,"",@progbits
	.quad	my_label_03-.Lline_table_start0

	.section	.offset_05,"",@progbits
	.quad	my_label_04-.Lline_table_start0
