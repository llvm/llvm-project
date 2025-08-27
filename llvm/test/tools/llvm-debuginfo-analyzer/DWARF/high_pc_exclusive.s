# Regression test for:
#  - Instructions at DW_AT_high_pc of a scope incorrectly included in the scope

# clang test.cpp --target=i686-pc-linux -g -O0
# 1
# 2 int main(void) {
# 3   float ret = 0;
# 4   for (int i = 0; i < 10; i++) {
# 5     ret += i;
# 6   }
# 7   return ret;
# 8 }
# 9

# RUN: llvm-mc %s -triple=i686-pc-linux -filetype=obj -o - | \
# RUN: llvm-debuginfo-analyzer --attribute=all \
# RUN:                         --print=all \
# RUN:                         --output-sort=offset \
# RUN:                         - | \
# RUN: FileCheck %s

# Make sure the line mapping at 0x3c does not show up at the scope level 004
# CHECK-NOT: [0x000000003c][004] 7 {Line}

# Make sure it *does* appear at scope level 003
# CHECK: [0x000000003c][003] 7 {Line}

	.file	"compile.cpp"
	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "test.cpp"
	.loc	0 2 0
	.cfi_startproc
# %bb.0:                                # %entry
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	subl	$12, %esp
	movl	$0, -4(%ebp)
.Ltmp0:
	.loc	0 3 15 prologue_end
	xorps	%xmm0, %xmm0
	movss	%xmm0, -8(%ebp)
.Ltmp1:
	.loc	0 4 18
	movl	$0, -12(%ebp)
.LBB0_1:                                # %for.cond
                                        # =>This Inner Loop Header: Depth=1
.Ltmp2:
	.loc	0 4 27 is_stmt 0
	cmpl	$10, -12(%ebp)
.Ltmp3:
	.loc	0 4 9
	jge	.LBB0_4
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
.Ltmp4:
	.loc	0 5 18 is_stmt 1
	cvtsi2ssl	-12(%ebp), %xmm0
	.loc	0 5 15 is_stmt 0
	addss	-8(%ebp), %xmm0
	movss	%xmm0, -8(%ebp)
.Ltmp5:
# %bb.3:                                # %for.inc
                                        #   in Loop: Header=BB0_1 Depth=1
	.loc	0 4 34 is_stmt 1
	movl	-12(%ebp), %eax
	addl	$1, %eax
	movl	%eax, -12(%ebp)
	.loc	0 4 9 is_stmt 0
	jmp	.LBB0_1
.Ltmp6:
.LBB0_4:                                # %for.end
	.loc	0 7 16 is_stmt 1
	cvttss2si	-8(%ebp), %eax
	.loc	0 7 9 epilogue_begin is_stmt 0
	addl	$12, %esp
	popl	%ebp
	.cfi_def_cfa %esp, 4
	retl
.Ltmp7:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x4d DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x2d DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	85
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	80                              # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x32:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	84                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x3d:0x12 DW_TAG_lexical_block
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp6-.Ltmp1                   # DW_AT_high_pc
	.byte	3                               # Abbrev [3] 0x43:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	80                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x50:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x54:0x4 DW_TAG_base_type
	.byte	6                               # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	36                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 22.0.0" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp" # string offset=113
.Linfo_string2:
	.asciz	"F:\\llvm-project"              # string offset=142
.Linfo_string3:
	.asciz	"main"                          # string offset=158
.Linfo_string4:
	.asciz	"int"                           # string offset=163
.Linfo_string5:
	.asciz	"ret"                           # string offset=167
.Linfo_string6:
	.asciz	"float"                         # string offset=171
.Linfo_string7:
	.asciz	"i"                             # string offset=177
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	4                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.long	.Lfunc_begin0
	.long	.Ltmp1
.Ldebug_addr_end0:
	.ident	"clang version 22.0.0"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
