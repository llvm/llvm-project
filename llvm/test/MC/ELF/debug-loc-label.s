// Verify that the .loc_label instruction resets the line sequence and generates
// the requested label at the correct position in the line stream

// RUN: llvm-mc -filetype obj -triple x86_64-linux-elf %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=CHECK-ASM
// RUN: llvm-dwarfdump -v --debug-line %t.o | FileCheck %s --check-prefix=CHECK-LINE-TABLE

# CHECK-ASM:        <foo>:
# CHECK-ASM-NEXT:     movq    %rdx, 0x33
# CHECK-ASM-NEXT:     movq    %rax, 0x3b
# CHECK-ASM-NEXT:     movq    %rbx, 0x4e
# CHECK-ASM-NEXT:     movq    %rcx, 0x60
# CHECK-ASM-NEXT:     retq

# CHECK-LINE-TABLE:                  Address            Line   Column File   ISA Discriminator OpIndex Flags
# CHECK-LINE-TABLE-NEXT:             ------------------ ------ ------ ------ --- ------------- ------- -------------
# CHECK-LINE-TABLE-NEXT: 0x00000028: 05 DW_LNS_set_column (1)
# CHECK-LINE-TABLE-NEXT: 0x0000002a: 00 DW_LNE_set_address (0x0000000000000000)
# CHECK-LINE-TABLE-NEXT: 0x00000035: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000000      1      1      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000036: 02 DW_LNS_advance_pc (addr += 33, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x00000038: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000021      1      1      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000003b: 05 DW_LNS_set_column (2)
# CHECK-LINE-TABLE-NEXT: 0x0000003d: 00 DW_LNE_set_address (0x0000000000000008)
# CHECK-LINE-TABLE-NEXT: 0x00000048: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000008      1      2      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000049: 02 DW_LNS_advance_pc (addr += 25, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x0000004b: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000021      1      2      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000004e: 05 DW_LNS_set_column (3)
# CHECK-LINE-TABLE-NEXT: 0x00000050: 00 DW_LNE_set_address (0x0000000000000010)
# CHECK-LINE-TABLE-NEXT: 0x0000005b: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000010      1      3      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x0000005c: 08 DW_LNS_const_add_pc (addr += 0x0000000000000011, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x0000005d: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000021      1      3      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x00000060: 05 DW_LNS_set_column (4)
# CHECK-LINE-TABLE-NEXT: 0x00000062: 00 DW_LNE_set_address (0x0000000000000018)
# CHECK-LINE-TABLE-NEXT: 0x0000006d: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      4      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x0000006e: 05 DW_LNS_set_column (5)
# CHECK-LINE-TABLE-NEXT: 0x00000070: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      5      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000071: 02 DW_LNS_advance_pc (addr += 9, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x00000073: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000021      1      5      1   0             0       0  is_stmt end_sequence
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
	mov     %rdx, 0x33
	.loc_label my_label_02
	.loc	1 1 2
  movq    %rax, my_label_02-.Lline_table_start0
	.loc	1 1 3
	.loc_label my_label_03
	.loc_label my_label_03.1
  movq    %rbx, my_label_03-.Lline_table_start0
	.loc	1 1 4
	.loc_label my_label_05
	.loc	1 1 5
  movq    %rcx, my_label_05-.Lline_table_start0
	ret
	.cfi_endproc

	.section	.debug_line,"",@progbits
.Lline_table_start0:
