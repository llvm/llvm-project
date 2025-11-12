/* Original C (for context):
* __attribute__((noinline))
* int regs_int_params(int a, int b, int c, int d, int e, int f) {
*   // Keep all params in regs; avoid spilling to the stack.
*   // The compiler will usually keep a..f in the 6 integer-arg regs.
*   asm volatile("" :: "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f));
*   return a + b + c + d + e + f;
* }
*/
	.file	"regs_int_params.c"
	.text
	.globl	regs_int_params                 # -- Begin function regs_int_params
	.p2align	4
	.type	regs_int_params,@function
regs_int_params:                        # @regs_int_params
.Lfunc_begin0:
	.file	0 "." "regs_int_params.c" md5 0xcf39432098ab893043cc8b4606354bd2
	.loc	0 2 0                           # regs_int_params.c:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: regs_int_params:a <- $edi
	#DEBUG_VALUE: regs_int_params:b <- $esi
	#DEBUG_VALUE: regs_int_params:c <- $edx
	#DEBUG_VALUE: regs_int_params:d <- $ecx
	#DEBUG_VALUE: regs_int_params:e <- $r8d
	#DEBUG_VALUE: regs_int_params:f <- $r9d
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
                                        # kill: def $r9d killed $r9d def $r9
                                        # kill: def $r8d killed $r8d def $r8
                                        # kill: def $ecx killed $ecx def $rcx
                                        # kill: def $edx killed $edx def $rdx
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
.Ltmp0:
	.loc	0 5 3 prologue_end              # regs_int_params.c:5:3
	#APP
	#NO_APP
	.loc	0 6 12                          # regs_int_params.c:6:12
	addl	%edi, %esi
.Ltmp1:
	#DEBUG_VALUE: regs_int_params:b <- [DW_OP_LLVM_entry_value 1] $esi
	.loc	0 6 16 is_stmt 0                # regs_int_params.c:6:16
	leal	(%rdx,%rcx), %eax
	.loc	0 6 20                          # regs_int_params.c:6:20
	addl	%esi, %eax
	.loc	0 6 28                          # regs_int_params.c:6:28
	addl	%r8d, %eax
	addl	%r9d, %eax
	.loc	0 6 3 epilogue_begin            # regs_int_params.c:6:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp2:
.Lfunc_end0:
	.size	regs_int_params, .Lfunc_end0-regs_int_params
	.cfi_endproc
                                        # -- End function
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    #   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   ending offset
	.byte	4                               # Loc expr size
	.byte	163                             # DW_OP_entry_value
	.byte	1                               # 1
	.byte	84                              # super-register DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
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
	.ascii	"\214\001"                      # DW_AT_loclists_base
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
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
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
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
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
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x6b DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	29                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lloclists_table_base0          # DW_AT_loclists_base
	.byte	2                               # Abbrev [2] 0x27:0x4b DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
                                        # DW_AT_call_all_calls
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	114                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x36:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x40:0x9 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_location
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x49:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x53:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	82
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x5d:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	88
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x67:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	89
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x72:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	48                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 22.0.0git (https://github.com/UltimateForce21/llvm-project.git 79c0a9e1e7da0f727c41d27c9c6ff8a28bb7d06f)" # string offset=0
.Linfo_string1:
	.asciz	"regs_int_params.c"             # string offset=119
.Linfo_string2:
	.asciz	"."                             # string offset=137
.Linfo_string3:
	.asciz	"regs_int_params"               # string offset=139
.Linfo_string4:
	.asciz	"int"                           # string offset=155
.Linfo_string5:
	.asciz	"a"                             # string offset=159
.Linfo_string6:
	.asciz	"b"                             # string offset=161
.Linfo_string7:
	.asciz	"c"                             # string offset=163
.Linfo_string8:
	.asciz	"d"                             # string offset=165
.Linfo_string9:
	.asciz	"e"                             # string offset=167
.Linfo_string10:
	.asciz	"f"                             # string offset=169
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 22.0.0git (https://github.com/UltimateForce21/llvm-project.git 79c0a9e1e7da0f727c41d27c9c6ff8a28bb7d06f)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
