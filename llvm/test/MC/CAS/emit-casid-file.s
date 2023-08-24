// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-native %s --mccas-emit-casid-file -o %t/test.o
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=NATIVE_FILENAME
// NATIVE_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-verify %s --mccas-emit-casid-file -o %t/test.o
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=VERIFY_FILENAME
// VERIFY_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-casid %s --mccas-emit-casid-file -o %t/test.o
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-native %s --mccas-emit-casid-file -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-verify %s --mccas-emit-casid-file -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=/tmp/cas --mccas-casid %s --mccas-emit-casid-file -o -
// RUN: not cat %t/test.o.casid

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	__Z3fooi                        ; -- Begin function _Z3fooi
	.p2align	2
__Z3fooi:                               ; @_Z3fooi
Lfunc_begin0:
	.file	1 "/Users/shubham/Development" "test109275485/a.cpp"
	.loc	1 1 0                           ; test109275485/a.cpp:1:0
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #12]
Ltmp1:
	.loc	1 2 12 prologue_end             ; test109275485/a.cpp:2:12
	ldr	w8, [sp, #12]
	.loc	1 2 13 is_stmt 0                ; test109275485/a.cpp:2:13
	add	w0, w8, #2
	.loc	1 2 5 epilogue_begin            ; test109275485/a.cpp:2:5
	add	sp, sp, #16
	ret
Ltmp2:
Lfunc_end0:
	.cfi_endproc
                                        ; -- End function
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.ascii	"\202|"                         ; DW_AT_LLVM_sysroot
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.ascii	"\347\177"                      ; DW_AT_APPLE_omit_frame_ptr
	.byte	25                              ; DW_FORM_flag_present
	.byte	64                              ; DW_AT_frame_base
	.byte	24                              ; DW_FORM_exprloc
	.byte	110                             ; DW_AT_linkage_name
	.byte	14                              ; DW_FORM_strp
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                               ; DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x57 DW_TAG_compile_unit
	.long	0                               ; DW_AT_producer
	.short	33                              ; DW_AT_language
	.long	102                             ; DW_AT_name
	.long	149                             ; DW_AT_LLVM_sysroot
.set Lset2, Lline_table_start0-Lsection_line ; DW_AT_stmt_list
	.long	Lset2
	.long	151                             ; DW_AT_comp_dir
	.quad	Lfunc_begin0                    ; DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ; DW_AT_high_pc
	.long	Lset3
	.byte	2                               ; Abbrev [2] 0x2e:0x2c DW_TAG_subprogram
	.quad	Lfunc_begin0                    ; DW_AT_low_pc
.set Lset4, Lfunc_end0-Lfunc_begin0     ; DW_AT_high_pc
	.long	Lset4
                                        ; DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ; DW_AT_frame_base
	.byte	111
	.long	224                             ; DW_AT_linkage_name
	.long	220                             ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.long	90                              ; DW_AT_type
                                        ; DW_AT_external
	.byte	3                               ; Abbrev [3] 0x4b:0xe DW_TAG_formal_parameter
	.byte	2                               ; DW_AT_location
	.byte	145
	.byte	12
	.long	236                             ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.long	90                              ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x5a:0x7 DW_TAG_base_type
	.long	232                             ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	4                               ; DW_AT_byte_size
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 18.0.0 (git@github.com:apple/llvm-project.git 7f16558429157c87cdbe8802086dd04b0deed7f7)" ; string offset=0
	.asciz	"/Users/shubham/Development/test109275485/a.cpp" ; string offset=102
	.asciz	"/"                             ; string offset=149
	.asciz	"/Users/shubham/Development/llvm-project-cas/llvm-project/build_ninja" ; string offset=151
	.asciz	"foo"                           ; string offset=220
	.asciz	"_Z3fooi"                       ; string offset=224
	.asciz	"int"                           ; string offset=232
	.asciz	"x"                             ; string offset=236
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	2                               ; Header Bucket Count
	.long	2                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	0                               ; Bucket 0
	.long	1                               ; Bucket 1
	.long	1784752350                      ; Hash in Bucket 0
	.long	193491849                       ; Hash in Bucket 1
.set Lset5, LNames1-Lnames_begin        ; Offset in Bucket 0
	.long	Lset5
.set Lset6, LNames0-Lnames_begin        ; Offset in Bucket 1
	.long	Lset6
LNames1:
	.long	224                             ; _Z3fooi
	.long	1                               ; Num DIEs
	.long	46
	.long	0
LNames0:
	.long	220                             ; foo
	.long	1                               ; Num DIEs
	.long	46
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	0                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	-1                              ; Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	0                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	-1                              ; Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	1                               ; Header Hash Count
	.long	20                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	3                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.short	3                               ; DW_ATOM_die_tag
	.short	5                               ; DW_FORM_data2
	.short	4                               ; DW_ATOM_type_flags
	.short	11                              ; DW_FORM_data1
	.long	0                               ; Bucket 0
	.long	193495088                       ; Hash in Bucket 0
.set Lset7, Ltypes0-Ltypes_begin        ; Offset in Bucket 0
	.long	Lset7
Ltypes0:
	.long	232                             ; int
	.long	1                               ; Num DIEs
	.long	90
	.short	36
	.byte	0
	.long	0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
