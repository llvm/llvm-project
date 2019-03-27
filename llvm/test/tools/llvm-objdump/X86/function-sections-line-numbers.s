# The code below is the reduced version of the output
# from the following invocation and source:
#
# // test.cpp:
#void f1() {}
#void f2() {}
#
# clang -gdwarf-5 -ffunction-sections test.cpp -o test.s -S

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux -dwarf-version=5 %s -o %t.o
# RUN: llvm-objdump -disassemble -line-numbers -r -s -section-headers -t %t.o | FileCheck %s


# CHECK: 0000000000000000 _Z2f1v
# CHECK-NOT: test.cpp:2
# CHECK: test.cpp:1
# CHECK-NOT: test.cpp:2
# CHECK: 0000000000000000 _Z2f2v
# CHECK-NOT: test.cpp:1
# CHECK: test.cpp:2
# CHECK-NOT: test.cpp:1


	.text
	.file	"test.cpp"
	.section	.text._Z2f1v,"ax",@progbits
	.globl	_Z2f1v                  # -- Begin function _Z2f1v
	.p2align	4, 0x90
	.type	_Z2f1v,@function
_Z2f1v:                                 # @_Z2f1v
.Lfunc_begin0:
	.file	0 "/home/avl" "test.cpp" md5 0xefae234cc05b45384d782316d3a5d338
	.file	1 "test.cpp" md5 0xefae234cc05b45384d782316d3a5d338
	.loc	1 1 0                   # test.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	1 1 12 prologue_end     # test.cpp:1:12
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f1v, .Lfunc_end0-_Z2f1v
	.cfi_endproc
                                        # -- End function
	.section	.text._Z2f2v,"ax",@progbits
	.globl	_Z2f2v                  # -- Begin function _Z2f2v
	.p2align	4, 0x90
	.type	_Z2f2v,@function
_Z2f2v:                                 # @_Z2f2v
.Lfunc_begin1:
	.loc	1 2 0                   # test.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp2:
	.loc	1 2 12 prologue_end     # test.cpp:2:12
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	_Z2f2v, .Lfunc_end1-_Z2f2v
	.cfi_endproc
                                        # -- End function
	.section	.debug_str_offsets,"",@progbits
	.long	32
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 9.0.0 (https://github.com/llvm/llvm-project.git ebfc1e5af7a65381d858612517e6414ef58df482)" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp"              # string offset=104
.Linfo_string2:
	.asciz	"/home/avl"             # string offset=113
.Linfo_string3:
	.asciz	"_Z2f1v"                # string offset=123
.Linfo_string4:
	.asciz	"f1"                    # string offset=130
.Linfo_string5:
	.asciz	"_Z2f2v"                # string offset=133
.Linfo_string6:
	.asciz	"f2"                    # string offset=140
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	37                      # DW_FORM_strx1
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	114                     # DW_AT_str_offsets_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	37                      # DW_FORM_strx1
	.byte	115                     # DW_AT_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	85                      # DW_AT_ranges
	.byte	35                      # DW_FORM_rnglistx
	.byte	116                     # DW_AT_rnglists_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	110                     # DW_AT_linkage_name
	.byte	37                      # DW_FORM_strx1
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] 0xc:0x38 DW_TAG_compile_unit
	.byte	0                       # DW_AT_producer
	.short	4                       # DW_AT_language
	.byte	1                       # DW_AT_name
	.long	.Lstr_offsets_base0     # DW_AT_str_offsets_base
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	2                       # DW_AT_comp_dir
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.quad	0                       # DW_AT_low_pc
	.byte	0                       # DW_AT_ranges
	.long	.Lrnglists_table_base0  # DW_AT_rnglists_base
	.byte	2                       # Abbrev [2] 0x2b:0xc DW_TAG_subprogram
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	86
	.byte	3                       # DW_AT_linkage_name
	.byte	4                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
                                        # DW_AT_external
	.byte	2                       # Abbrev [2] 0x37:0xc DW_TAG_subprogram
	.byte	1                       # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	86
	.byte	5                       # DW_AT_linkage_name
	.byte	6                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	2                       # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_rnglist_table_end0-.Ldebug_rnglist_table_start0 # Length
.Ldebug_rnglist_table_start0:
	.short	5                       # Version
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
	.long	1                       # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3                       # DW_RLE_startx_length
	.byte	0                       #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0 #   length
	.byte	3                       # DW_RLE_startx_length
	.byte	1                       #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1 #   length
	.byte	0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end0:
.Ldebug_addr_start0:
	.short	5                       # DWARF version number
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:

	.section	.debug_line,"",@progbits
.Lline_table_start0:
