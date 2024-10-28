	.text
	.file	"ThreadLocalStorage.cpp"
	.file	0 "/data/projects/sandbox/debuginfo-analyzer" "ThreadLocalStorage.cpp" md5 0x8fc7af759be43d1ae3bc8c835ed18e5b
	.globl	_Z4testv                        # -- Begin function _Z4testv
	.p2align	4
	.type	_Z4testv,@function
_Z4testv:                               # @_Z4testv
.Lfunc_begin0:
	.loc	0 3 0                           # ThreadLocalStorage.cpp:3:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 5 3 prologue_end              # ThreadLocalStorage.cpp:5:3
	movq	%fs:0, %rax
	leaq	TGlobal@TPOFF(%rax), %rax
	.loc	0 5 11 is_stmt 0                # ThreadLocalStorage.cpp:5:11
	movl	$1, (%rax)
	.loc	0 7 7 is_stmt 1                 # ThreadLocalStorage.cpp:7:7
	movl	$0, -4(%rbp)
	.loc	0 8 11                          # ThreadLocalStorage.cpp:8:11
	movl	$2, NGlobal(%rip)
	.loc	0 9 1 epilogue_begin            # ThreadLocalStorage.cpp:9:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z4testv, .Lfunc_end0-_Z4testv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZTW7TGlobal,"axG",@progbits,_ZTW7TGlobal,comdat
	.hidden	_ZTW7TGlobal                    # -- Begin function _ZTW7TGlobal
	.weak	_ZTW7TGlobal
	.p2align	4
	.type	_ZTW7TGlobal,@function
_ZTW7TGlobal:                           # @_ZTW7TGlobal
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%fs:0, %rax
	leaq	TGlobal@TPOFF(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_ZTW7TGlobal, .Lfunc_end1-_ZTW7TGlobal
	.cfi_endproc
                                        # -- End function
	.type	TGlobal,@object                 # @TGlobal
	.section	.tbss,"awT",@nobits
	.globl	TGlobal
	.p2align	2, 0x0
TGlobal:
	.long	0                               # 0x0
	.size	TGlobal, 4

	.type	NGlobal,@object                 # @NGlobal
	.data
	.globl	NGlobal
	.p2align	2, 0x0
NGlobal:
	.long	1                               # 0x1
	.size	NGlobal, 4

	.type	_ZZ4testvE6TLocal,@object       # @_ZZ4testvE6TLocal
	.section	.tbss,"awT",@nobits
	.p2align	2, 0x0
_ZZ4testvE6TLocal:
	.long	0                               # 0x0
	.size	_ZZ4testvE6TLocal, 4

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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
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
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x65 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x13 DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	10                              # DW_AT_location
	.byte	14
	.quad	TGlobal@DTPOFF
	.byte	224
	.byte	3                               # Abbrev [3] 0x36:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x3a:0xb DW_TAG_variable
	.byte	5                               # DW_AT_name
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	4                               # Abbrev [4] 0x45:0x2b DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	7                               # DW_AT_linkage_name
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x51:0x13 DW_TAG_variable
	.byte	6                               # DW_AT_name
	.long	54                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	10                              # DW_AT_location
	.byte	14
	.quad	_ZZ4testvE6TLocal@DTPOFF
	.byte	224
	.byte	6                               # Abbrev [6] 0x64:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	44                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 20.0.0git (https://github.com/CarlosAlbertoEnciso/llvm-project.git 38fca7b7db2ba1647c87679d6750fc4a4bfe72e1)" # string offset=0
.Linfo_string1:
	.asciz	"ThreadLocalStorage.cpp"        # string offset=123
.Linfo_string2:
	.asciz	"/data/projects/sandbox/debuginfo-analyzer" # string offset=146
.Linfo_string3:
	.asciz	"TGlobal"                       # string offset=188
.Linfo_string4:
	.asciz	"int"                           # string offset=196
.Linfo_string5:
	.asciz	"NGlobal"                       # string offset=200
.Linfo_string6:
	.asciz	"TLocal"                        # string offset=208
.Linfo_string7:
	.asciz	"_Z4testv"                      # string offset=215
.Linfo_string8:
	.asciz	"test"                          # string offset=224
.Linfo_string9:
	.asciz	"NLocal"                        # string offset=229
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	NGlobal
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 20.0.0git (https://github.com/CarlosAlbertoEnciso/llvm-project.git 38fca7b7db2ba1647c87679d6750fc4a4bfe72e1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym NGlobal
	.section	.debug_line,"",@progbits
.Lline_table_start0:
