	.build_version macos, 26, 0	sdk_version 26, 4
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_test                           ; -- Begin function test
	.p2align	2
_test:                                  ; @test
Lfunc_begin0:
	.file	0 "." "macho-profile.c" md5 0xb946e1f4a94b0a399ee64ce179614690
	.cfi_startproc
; %bb.0:
	;DEBUG_VALUE: test:lo <- $w0
	;DEBUG_VALUE: test:hi <- $w1
	;DEBUG_VALUE: test:out <- $x2
	;DEBUG_VALUE: test:sum <- 0
	.loc	0 3 3 prologue_end              ; macho-profile.c:3:3
	cmp	w0, w1
	b.ge	LBB0_3
Ltmp0:
; %bb.1:
	;DEBUG_VALUE: test:sum <- 0
	;DEBUG_VALUE: test:out <- $x2
	;DEBUG_VALUE: test:hi <- $w1
	;DEBUG_VALUE: test:lo <- $w0
	.loc	0 0 3 is_stmt 0                 ; macho-profile.c:0:3
	mov	w8, #0                          ; =0x0
Ltmp1:
LBB0_2:                                 ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: test:out <- $x2
	;DEBUG_VALUE: test:hi <- $w1
	;DEBUG_VALUE: test:lo <- $w0
	;DEBUG_VALUE: test:lo <- $w0
	;DEBUG_VALUE: test:sum <- $w8
	.loc	0 4 9 is_stmt 1                 ; macho-profile.c:4:9
	add	w8, w8, w0
Ltmp2:
	;DEBUG_VALUE: test:sum <- $w8
	.loc	0 5 8                           ; macho-profile.c:5:8
	lsl	w0, w0, #1
Ltmp3:
	;DEBUG_VALUE: test:lo <- $w0
	.loc	0 3 3                           ; macho-profile.c:3:3
	cmp	w0, w1
	b.lt	LBB0_2
	b	LBB0_4
Ltmp4:
LBB0_3:
	;DEBUG_VALUE: test:sum <- 0
	;DEBUG_VALUE: test:out <- $x2
	;DEBUG_VALUE: test:hi <- $w1
	;DEBUG_VALUE: test:lo <- $w0
	.loc	0 0 3 is_stmt 0                 ; macho-profile.c:0:3
	mov	w8, #0                          ; =0x0
Ltmp5:
LBB0_4:
	;DEBUG_VALUE: test:out <- $x2
	;DEBUG_VALUE: test:hi <- $w1
	;DEBUG_VALUE: test:lo <- $w0
	.loc	0 7 8 is_stmt 1                 ; macho-profile.c:7:8
	str	w8, [x2]
	.loc	0 8 3                           ; macho-profile.c:8:3
	mov	x0, x8
Ltmp6:
	ret
Ltmp7:
Lfunc_end0:
	.cfi_endproc
                                        ; -- End function
	.section	__DWARF,__debug_loclists,regular,debug
Lsection_debug_loc0:
Lset0 = Ldebug_list_header_end0-Ldebug_list_header_start0 ; Length
	.long	Lset0
Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	2                               ; Offset entry count
Lloclists_table_base0:
Lset1 = Ldebug_loc0-Lloclists_table_base0
	.long	Lset1
Lset2 = Ldebug_loc1-Lloclists_table_base0
	.long	Lset2
Ldebug_loc0:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin0-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp6-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	80                              ; DW_OP_reg0
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_loc1:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Lfunc_begin0-Lfunc_begin0      ;   starting offset
	.uleb128 Ltmp1-Lfunc_begin0             ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	0                               ; 0
	.byte	159                             ; DW_OP_stack_value
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp1-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp4-Lfunc_begin0             ;   ending offset
	.byte	1                               ; Loc expr size
	.byte	88                              ; DW_OP_reg8
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 Ltmp4-Lfunc_begin0             ;   starting offset
	.uleb128 Ltmp5-Lfunc_begin0             ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	17                              ; DW_OP_consts
	.byte	0                               ; 0
	.byte	159                             ; DW_OP_stack_value
	.byte	0                               ; DW_LLE_end_of_list
Ldebug_list_header_end0:
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\202|"                         ; DW_AT_LLVM_sysroot
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\357\177"                      ; DW_AT_APPLE_sdk
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.ascii	"\214\001"                      ; DW_AT_loclists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.ascii	"\347\177"                      ; DW_AT_APPLE_omit_frame_ptr
	.byte	25                              ; DW_FORM_flag_present
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	39                              ; DW_AT_prototyped
	.byte	25                              ; DW_FORM_flag_present
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset3 = Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset3
Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
Lset4 = Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset4
	.byte	1                               ; Abbrev [1] 0xc:0x5d DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	29                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.byte	2                               ; DW_AT_LLVM_sysroot
	.byte	3                               ; DW_AT_APPLE_sdk
Lset5 = Lstr_offsets_base0-Lsection_str_off ; DW_AT_str_offsets_base
	.long	Lset5
Lset6 = Lline_table_start0-Lsection_line ; DW_AT_stmt_list
	.long	Lset6
	.byte	4                               ; DW_AT_comp_dir
                                        ; DW_AT_APPLE_optimized
	.byte	0                               ; DW_AT_low_pc
Lset7 = Lfunc_end0-Lfunc_begin0         ; DW_AT_high_pc
	.long	Lset7
Lset8 = Laddr_table_base0-Lsection_info0 ; DW_AT_addr_base
	.long	Lset8
Lset9 = Lloclists_table_base0-Lsection_debug_loc0 ; DW_AT_loclists_base
	.long	Lset9
	.byte	2                               ; Abbrev [2] 0x29:0x36 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
Lset10 = Lfunc_end0-Lfunc_begin0        ; DW_AT_high_pc
	.long	Lset10
                                        ; DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ; DW_AT_frame_base
	.byte	111
                                        ; DW_AT_call_all_calls
	.byte	5                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
                                        ; DW_AT_prototyped
	.long	95                              ; DW_AT_type
                                        ; DW_AT_external
                                        ; DW_AT_APPLE_optimized
	.byte	3                               ; Abbrev [3] 0x38:0x9 DW_TAG_formal_parameter
	.byte	0                               ; DW_AT_location
	.byte	7                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.long	95                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x41:0xa DW_TAG_formal_parameter
	.byte	1                               ; DW_AT_location
	.byte	81
	.byte	8                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.long	95                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x4b:0xa DW_TAG_formal_parameter
	.byte	1                               ; DW_AT_location
	.byte	82
	.byte	9                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.long	99                              ; DW_AT_type
	.byte	5                               ; Abbrev [5] 0x55:0x9 DW_TAG_variable
	.byte	1                               ; DW_AT_location
	.byte	10                              ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	2                               ; DW_AT_decl_line
	.long	95                              ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x5f:0x4 DW_TAG_base_type
	.byte	6                               ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	4                               ; DW_AT_byte_size
	.byte	7                               ; Abbrev [7] 0x63:0x5 DW_TAG_pointer_type
	.long	95                              ; DW_AT_type
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	48                              ; Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"Apple clang version 21.0.0 (clang-2100.0.123.102)" ; string offset=0
	.asciz	"macho-profile.c"               ; string offset=50
	.asciz	"/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.4.sdk" ; string offset=66
	.asciz	"MacOSX26.4.sdk"                ; string offset=165
	.asciz	"."                             ; string offset=180
	.asciz	"test"                          ; string offset=182
	.asciz	"int"                           ; string offset=187
	.asciz	"lo"                            ; string offset=191
	.asciz	"hi"                            ; string offset=194
	.asciz	"out"                           ; string offset=197
	.asciz	"sum"                           ; string offset=201
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	50
	.long	66
	.long	165
	.long	180
	.long	182
	.long	187
	.long	191
	.long	194
	.long	197
	.long	201
	.section	__DWARF,__debug_addr,regular,debug
Lsection_info0:
Lset11 = Ldebug_addr_end0-Ldebug_addr_start0 ; Length of contribution
	.long	Lset11
Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
Laddr_table_base0:
	.quad	Lfunc_begin0
Ldebug_addr_end0:
	.section	__DWARF,__debug_names,regular,debug
Ldebug_names_begin:
Lset12 = Lnames_end0-Lnames_start0      ; Header: unit length
	.long	Lset12
Lnames_start0:
	.short	5                               ; Header: version
	.short	0                               ; Header: padding
	.long	1                               ; Header: compilation unit count
	.long	0                               ; Header: local type unit count
	.long	0                               ; Header: foreign type unit count
	.long	2                               ; Header: bucket count
	.long	2                               ; Header: name count
Lset13 = Lnames_abbrev_end0-Lnames_abbrev_start0 ; Header: abbreviation table size
	.long	Lset13
	.long	8                               ; Header: augmentation string size
	.ascii	"LLVM0700"                      ; Header: augmentation string
Lset14 = Lcu_begin0-Lsection_info       ; Compilation unit 0
	.long	Lset14
	.long	1                               ; Bucket 0
	.long	2                               ; Bucket 1
	.long	193495088                       ; Hash in Bucket 0
	.long	2090756197                      ; Hash in Bucket 1
	.long	187                             ; String in Bucket 0: int
	.long	182                             ; String in Bucket 1: test
Lset15 = Lnames1-Lnames_entries0        ; Offset in Bucket 0
	.long	Lset15
Lset16 = Lnames0-Lnames_entries0        ; Offset in Bucket 1
	.long	Lset16
Lnames_abbrev_start0:
	.byte	1                               ; Abbrev code
	.byte	36                              ; DW_TAG_base_type
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.byte	2                               ; Abbrev code
	.byte	46                              ; DW_TAG_subprogram
	.byte	3                               ; DW_IDX_die_offset
	.byte	19                              ; DW_FORM_ref4
	.byte	4                               ; DW_IDX_parent
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev
	.byte	0                               ; End of abbrev list
Lnames_abbrev_end0:
Lnames_entries0:
Lnames1:
L1:
	.byte	1                               ; Abbreviation code
	.long	95                              ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: int
Lnames0:
L0:
	.byte	2                               ; Abbreviation code
	.long	41                              ; DW_IDX_die_offset
	.byte	0                               ; DW_IDX_parent
                                        ; End of list: test
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
