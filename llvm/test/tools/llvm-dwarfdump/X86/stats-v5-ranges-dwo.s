# RUN: rm -rf %t && mkdir -p %t
# RUN: cd %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux-gnu %s -o ranges.o -split-dwarf-file ranges.dwo
# RUN: llvm-dwarfdump --statistics ranges.o | FileCheck %s

# CHECK: "#bytes within functions": 30,


	.text
	.file	"ranges.cpp"
	.section	.text._Z2f3v,"ax",@progbits
	.globl	_Z2f3v                          # -- Begin function _Z2f3v
	.p2align	4, 0x90
	.type	_Z2f3v,@function
_Z2f3v:                                 # @_Z2f3v
.Lfunc_begin0:
	.file	0 "/usr/local/google/home/blaikie/dev/scratch" "ranges.cpp" md5 0x6dd27b2681f4bda8fe888ef8f6ffca94
	.loc	0 4 0                           # ranges.cpp:4:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 5 7 prologue_end              # ranges.cpp:5:7
	testb	$1, b
	je	_Z2f3v.__part.2
	jmp	_Z2f3v.__part.1
.LBB_END0_0:
	.cfi_endproc
	.section	.text._Z2f3v,"ax",@progbits,unique,1
_Z2f3v.__part.1:                        # %if.then
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 6 5                           # ranges.cpp:6:5
	callq	_Z2f1v
	jmp	_Z2f3v.__part.2
.LBB_END0_1:
	.size	_Z2f3v.__part.1, .LBB_END0_1-_Z2f3v.__part.1
	.cfi_endproc
	.section	.text._Z2f3v,"ax",@progbits,unique,2
_Z2f3v.__part.2:                        # %if.end
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 7 3                           # ranges.cpp:7:3
	callq	_Z2f2v
	.loc	0 8 1                           # ranges.cpp:8:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END0_2:
	.size	_Z2f3v.__part.2, .LBB_END0_2-_Z2f3v.__part.2
	.cfi_endproc
	.section	.text._Z2f3v,"ax",@progbits
.Lfunc_end0:
	.size	_Z2f3v, .Lfunc_end0-_Z2f3v
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	-7949078943903054930
	.byte	1                               # Abbrev [1] 0x14:0x1c DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	1                               # DW_AT_dwo_name
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges1:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .LBB_END0_1-_Z2f3v.__part.1    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .LBB_END0_2-_Z2f3v.__part.2    #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	2                               #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	12                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"ranges.dwo"                    # string offset=43
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string1
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	_Z2f3v.__part.1
	.quad	_Z2f3v.__part.2
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	26                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"f3"                            # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 13.0.0 (git@github.com:llvm/llvm-project.git 09cdcf09b54d328fc0a247b3a0f351d2610e928f)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z2f1v
	.addrsig_sym _Z2f2v
	.addrsig_sym b
	.section	.debug_line,"",@progbits
.Lline_table_start0:


	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	24                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z2f3v"                        # string offset=0
.Linfo_string1:
	.asciz	"f3"                            # string offset=7
.Linfo_string2:
	.asciz	"clang version 13.0.0 (git@github.com:llvm/llvm-project.git 09cdcf09b54d328fc0a247b3a0f351d2610e928f)" # string offset=10
.Linfo_string3:
	.asciz	"ranges.cpp"                    # string offset=111
.Linfo_string4:
	.asciz	"ranges.dwo"                    # string offset=122
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	7
	.long	10
	.long	111
	.long	122
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-7949078943903054930
	.byte	1                               # Abbrev [1] 0x14:0xf DW_TAG_compile_unit
	.byte	2                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	3                               # DW_AT_name
	.byte	4                               # DW_AT_dwo_name
	.byte	2                               # Abbrev [2] 0x1a:0x8 DW_TAG_subprogram
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_rnglists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 # Length
.Ldebug_list_header_start1:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_dwo_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 10                             #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 10                             #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	2                               #   start index
	.uleb128 10                             #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end1:

