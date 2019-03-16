# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer 0x0 0x2 0x3 0x4 --obj=%t.o | FileCheck %s
# This test makes sure we don't attempt to access out of the line table boundaries
# if the last range is empty.
# Produced from the following program:
# int func(int a) {
#   return 1 + a;
# }
# compiled with clang -O3 -g -S
# Edited by adding a redundant, empty last range.
# The line table (llvm-dwarfdump --debug-line) looks like:
#
# Address            Line   Column File   ISA Discriminator Flags
# ------------------ ------ ------ ------ --- ------------- -------------
# 0x0000000000000000      1      0      1   0             0  is_stmt
# 0x0000000000000000      2     12      1   0             0  is_stmt prologue_end
# 0x0000000000000003      2      3      1   0             0
# 0x0000000000000003      3      3      1   0             0
# 0x0000000000000004      3      3      1   0             0  end_sequence
#
# 0x0 should pick the second line in the table - line 2, col 12
# CHECK:    func
# CHECK:	/scratch/a.cpp:2:12
# 0x2 does not exist in the table, so we should pick the line with a
# lower address that describes the range - in this case, the second line.
# CHECK:    func
# CHECK:	/scratch/a.cpp:2:12
# 0x3 also has a sequence of empty ranges, so we should pick the first range after
# skipping the empty ones.
# This also verifies we don't attempt to access outside boundaries.
# CHECK:    func
# CHECK:	/scratch/a.cpp:3:3
# CHECK: ??
# CHECK: ??:0:0

	.text
	.file	"a.cpp"
	.globl	_Z4funci                # -- Begin function _Z4funci
	.p2align	4, 0x90
	.type	_Z4funci,@function
_Z4funci:                               # @_Z4funci
.Lfunc_begin0:
	.file	1 "/llvm-project" "/scratch/a.cpp"
	.loc	1 1 0                   # /scratch/a.cpp:1:0
	.cfi_startproc
# %bb.0:
	#DEBUG_VALUE: func:a <- $edi
                                        # kill: def $edi killed $edi def $rdi
	#DEBUG_VALUE: func:a <- $edi
	.loc	1 2 12 prologue_end     # /scratch/a.cpp:2:12
	leal	1(%rdi), %eax
	.loc	1 2 3                   # /scratch/a.cpp:2:3
    .loc    1 3 3 is_stmt 0         # this forms an empty range torgether with the previous.
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_Z4funci, .Lfunc_end0-_Z4funci
	.cfi_endproc
                                        # -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 7.0.1-6 (tags/RELEASE_701/final)" # string offset=0
.Linfo_string1:
	.asciz	"/scratch/a.cpp" # string offset=47
.Linfo_string2:
	.asciz	"/llvm-project" # string offset=97
.Linfo_string3:
	.asciz	"_Z4funci"              # string offset=146
.Linfo_string4:
	.asciz	"func"                  # string offset=155
.Linfo_string5:
	.asciz	"int"                   # string offset=160
.Linfo_string6:
	.asciz	"a"                     # string offset=164
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.ascii	"\264B"                 # DW_AT_GNU_pubnames
	.byte	25                      # DW_FORM_flag_present
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	91                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x54 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	4                       # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x2d DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string3          # DW_AT_linkage_name
	.long	.Linfo_string4          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.long	87                      # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x47:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.long	87                      # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	4                       # Abbrev [4] 0x57:0x7 DW_TAG_base_type
	.long	.Linfo_string5          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	95                      # Compilation Unit Length
	.long	42                      # DIE offset
	.asciz	"func"                  # External Name
	.long	0                       # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	95                      # Compilation Unit Length
	.long	87                      # DIE offset
	.asciz	"int"                   # External Name
	.long	0                       # End Mark
.LpubTypes_end0:

	.ident	"clang version 7.0.1-6 (tags/RELEASE_701/final)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
