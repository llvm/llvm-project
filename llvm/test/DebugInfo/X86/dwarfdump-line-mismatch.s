# The .debug_info section says 8-byte addresses, but the assembler
# will generate a line table with 4-byte addresses (for i686).

# RUN: llvm-mc -filetype=obj -triple i686-linux-gnu %s -o - | \
# RUN: llvm-dwarfdump -debug-line - 2>&1 | FileCheck %s

# CHECK: Mismatching address size at offset 0x{{[0-9a-f]+}}
# CHECK-SAME: expected 0x08 found 0x04
	.text
	.file	"reduced.c"
	.globl	main
main:
	.file	1 "/tmp" "reduced.c"
	.loc	1 2 0
	xorl	%eax, %eax
	retl
	.file	2 "/tmp/repeat/repeat/repeat/repeat" "repeat.h"

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

        .section	.debug_info,"",@progbits
	.long	.Lend0 - .Lbegin0       # Length of Unit
.Lbegin0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.debug_line             # DW_AT_stmt_list
.Lend0:
	.section	.debug_line,"",@progbits
