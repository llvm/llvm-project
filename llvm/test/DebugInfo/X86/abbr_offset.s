# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o - | \
# RUN: llvm-dwarfdump - | FileCheck %s

# This test simulates the result of ld -r. That produces files where
# abbr_offset is not zero.

# CHECK: abbr_offset = 0x0000
# CHECK: abbr_offset = 0x0008

       	.section	.debug_abbrev,"",@progbits
.Labbrev1:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

.Labbrev2:
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
	.long	.Labbrev1               # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Lline_table_start0     # DW_AT_stmt_list
.Lend0:

       	.long	.Lend1 - .Lbegin1       # Length of Unit
.Lbegin1:
	.short	4                       # DWARF version number
	.long	.Labbrev2               # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Lline_table_start0     # DW_AT_stmt_list
.Lend1:

	.section	.debug_line,"",@progbits
.Lline_table_start0:
