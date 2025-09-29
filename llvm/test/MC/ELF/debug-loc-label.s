// Verify that the .loc_label instruction resets the line sequence and generates
// the requested label at the correct position in the line stream

// RUN: llvm-mc -filetype obj -triple x86_64 %s -o %t.o
// RUN: llvm-dwarfdump -v --debug-line %t.o | FileCheck %s --check-prefix=CHECK-LINE-TABLE
// RUN: llvm-readelf -s %t.o | FileCheck %s --check-prefix=CHECK-SYM
// RUN: llvm-objdump -s -j .offsets %t.o | FileCheck %s --check-prefix=CHECK-OFFSETS

// RUN: not llvm-mc -filetype obj -triple x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
// RUN: not llvm-mc -filetype obj -triple x86_64 --defsym ERR2=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:



# CHECK-LINE-TABLE:                  Address            Line   Column File   ISA Discriminator OpIndex Flags
# CHECK-LINE-TABLE-NEXT:             ------------------ ------ ------ ------ --- ------------- ------- -------------
# CHECK-LINE-TABLE-NEXT: 0x00000028: 05 DW_LNS_set_column (1)
# CHECK-LINE-TABLE-NEXT: 0x0000002a: 00 DW_LNE_set_address (0x0000000000000000)
# CHECK-LINE-TABLE-NEXT: 0x00000035: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000000      1      1      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000036: 02 DW_LNS_advance_pc (addr += 8, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x00000038: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000008      1      1      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000003b: 05 DW_LNS_set_column (2)
# CHECK-LINE-TABLE-NEXT: 0x0000003d: 00 DW_LNE_set_address (0x0000000000000008)
# CHECK-LINE-TABLE-NEXT: 0x00000048: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000008      1      2      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000049: 02 DW_LNS_advance_pc (addr += 8, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x0000004b: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000010      1      2      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x0000004e: 05 DW_LNS_set_column (3)
# CHECK-LINE-TABLE-NEXT: 0x00000050: 00 DW_LNE_set_address (0x0000000000000010)
# CHECK-LINE-TABLE-NEXT: 0x0000005b: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000010      1      3      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x0000005c: 02 DW_LNS_advance_pc (addr += 8, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x0000005e: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      3      1   0             0       0  is_stmt end_sequence
# CHECK-LINE-TABLE-NEXT: 0x00000061: 05 DW_LNS_set_column (4)
# CHECK-LINE-TABLE-NEXT: 0x00000063: 00 DW_LNE_set_address (0x0000000000000018)
# CHECK-LINE-TABLE-NEXT: 0x0000006e: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      4      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x0000006f: 05 DW_LNS_set_column (5)
# CHECK-LINE-TABLE-NEXT: 0x00000071: 01 DW_LNS_copy
# CHECK-LINE-TABLE-NEXT:             0x0000000000000018      1      5      1   0             0       0  is_stmt
# CHECK-LINE-TABLE-NEXT: 0x00000072: 02 DW_LNS_advance_pc (addr += 8, op-index += 0)
# CHECK-LINE-TABLE-NEXT: 0x00000074: 00 DW_LNE_end_sequence
# CHECK-LINE-TABLE-NEXT:             0x0000000000000020      1      5      1   0             0       0  is_stmt end_sequence

# CHECK-SYM:      Symbol table '.symtab' contains 9 entries:
# CHECK-SYM-NEXT:    Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-SYM-NEXT:      0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-SYM-NEXT:      1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS test.c
# CHECK-SYM-NEXT:      2: 0000000000000000     0 SECTION LOCAL  DEFAULT     2 .text
# CHECK-SYM-NEXT:      3: 000000000000003b     0 NOTYPE  LOCAL  DEFAULT     3 my_label_02
# CHECK-SYM-NEXT:      4: 000000000000004e     0 NOTYPE  LOCAL  DEFAULT     3 my_label_03
# CHECK-SYM-NEXT:      5: 0000000000000061     0 NOTYPE  LOCAL  DEFAULT     3 my_label_04
# CHECK-SYM-NEXT:      6: 000000000000004e     0 NOTYPE  LOCAL  DEFAULT     3 my_label_03.1
# CHECK-SYM-NEXT:      7: 0000000000000077     0 NOTYPE  LOCAL  DEFAULT     3 my_label_05
# CHECK-SYM-NEXT:      8: 0000000000000000     0 FUNC    GLOBAL DEFAULT     2 foo

# CHECK-OFFSETS: 0000 3b000000 4e000000 61000000

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
.ifdef ERR
  .loc_label my_label_04
# ERR: [[#@LINE+1]]:13: error: expected identifier
  .loc_label
# ERR: [[#@LINE+1]]:19: error: expected newline
  .loc_label aaaa bbbb
.endif
.ifdef ERR2
# ERR2: [[#@LINE+1]]:14: error: symbol 'my_label_04' is already defined
  .loc_label my_label_04
.endif
	mov     %rax, 0x04
	.loc_label my_label_05
	ret
	.cfi_endproc

	.section	.debug_line,"",@progbits
.Lline_table_start0:

	.section	.offsets,"",@progbits
	.long	my_label_02-.Lline_table_start0
	.long	my_label_03-.Lline_table_start0
	.long	my_label_04-.Lline_table_start0
