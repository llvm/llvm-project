##; Generated with llc -O0 from the following IR, which mimics a C++ program
##; compiled with prefix mapping:
##;
##; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
##; target triple = "x86_64-apple-macosx10.15.0"
##;
##; %struct.Foo = type { i32 }
##;
##; define i32 @main() !dbg !20 {
##; entry:
##;   %f = alloca %struct.Foo, align 4
##;     #dbg_declare(ptr %f, !24, !DIExpression(), !23)
##;   ret i32 0, !dbg !23
##; }
##;
##; !llvm.dbg.cu = !{!0}
##; !llvm.module.flags = !{!30, !31}
##;
##; !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, sysroot: "/^sdk", sdk: "MacOSX.sdk")
##; !1 = !DIFile(filename: "main.cpp", directory: "/^src")
##; !10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !11, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !12, identifier: "_ZTS3Foo")
##; !11 = !DIFile(filename: "foo.h", directory: "/^src/include")
##; !12 = !{!13}
##; !13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !11, line: 1, baseType: !14, size: 32)
##; !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
##; !20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !21, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
##; !21 = !DISubroutineType(types: !22)
##; !22 = !{!14}
##; !23 = !DILocation(line: 4, column: 7, scope: !20)
##; !24 = !DILocalVariable(name: "f", scope: !20, file: !1, line: 4, type: !10)
##; !30 = !{i32 7, !"Dwarf Version", i32 5}
##; !31 = !{i32 2, !"Debug Info Version", i32 3}

	.build_version macos, 10, 15
	.section	__TEXT,__text,regular,pure_instructions
	.att_syntax
	.globl	_main                           ## -- Begin function main
	.p2align	4
_main:                                  ## @main
Lfunc_begin0:
	.file	0 "/^src" "main.cpp"
	.cfi_startproc
## %bb.0:                               ## %entry
	.loc	0 4 7 prologue_end              ## main.cpp:4:7
	xorl	%eax, %eax
	retq
Ltmp0:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.file	1 "/^src/include" "foo.h"
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ## Abbreviation Code
	.byte	17                              ## DW_TAG_compile_unit
	.byte	1                               ## DW_CHILDREN_yes
	.byte	37                              ## DW_AT_producer
	.byte	37                              ## DW_FORM_strx1
	.byte	19                              ## DW_AT_language
	.byte	5                               ## DW_FORM_data2
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.ascii	"\202|"                         ## DW_AT_LLVM_sysroot
	.byte	37                              ## DW_FORM_strx1
	.ascii	"\357\177"                      ## DW_AT_APPLE_sdk
	.byte	37                              ## DW_FORM_strx1
	.byte	114                             ## DW_AT_str_offsets_base
	.byte	23                              ## DW_FORM_sec_offset
	.byte	16                              ## DW_AT_stmt_list
	.byte	23                              ## DW_FORM_sec_offset
	.byte	27                              ## DW_AT_comp_dir
	.byte	37                              ## DW_FORM_strx1
	.byte	17                              ## DW_AT_low_pc
	.byte	27                              ## DW_FORM_addrx
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	115                             ## DW_AT_addr_base
	.byte	23                              ## DW_FORM_sec_offset
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	2                               ## Abbreviation Code
	.byte	46                              ## DW_TAG_subprogram
	.byte	1                               ## DW_CHILDREN_yes
	.byte	17                              ## DW_AT_low_pc
	.byte	27                              ## DW_FORM_addrx
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.ascii	"\347\177"                      ## DW_AT_APPLE_omit_frame_ptr
	.byte	25                              ## DW_FORM_flag_present
	.byte	64                              ## DW_AT_frame_base
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	3                               ## Abbreviation Code
	.byte	52                              ## DW_TAG_variable
	.byte	0                               ## DW_CHILDREN_no
	.byte	2                               ## DW_AT_location
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	4                               ## Abbreviation Code
	.byte	36                              ## DW_TAG_base_type
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	62                              ## DW_AT_encoding
	.byte	11                              ## DW_FORM_data1
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	5                               ## Abbreviation Code
	.byte	19                              ## DW_TAG_structure_type
	.byte	1                               ## DW_CHILDREN_yes
	.byte	54                              ## DW_AT_calling_convention
	.byte	11                              ## DW_FORM_data1
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	6                               ## Abbreviation Code
	.byte	13                              ## DW_TAG_member
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	56                              ## DW_AT_data_member_location
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	0                               ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	5                               ## DWARF version number
	.byte	1                               ## DWARF Unit Type
	.byte	8                               ## Address Size (in bytes)
Lset1 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	1                               ## Abbrev [1] 0xc:0x49 DW_TAG_compile_unit
	.byte	0                               ## DW_AT_producer
	.short	33                              ## DW_AT_language
	.byte	1                               ## DW_AT_name
	.byte	2                               ## DW_AT_LLVM_sysroot
	.byte	3                               ## DW_AT_APPLE_sdk
Lset2 = Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
	.long	Lset2
Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset3
	.byte	4                               ## DW_AT_comp_dir
	.byte	0                               ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
Lset5 = Laddr_table_base0-Lsection_info0 ## DW_AT_addr_base
	.long	Lset5
	.byte	2                               ## Abbrev [2] 0x25:0x1b DW_TAG_subprogram
	.byte	0                               ## DW_AT_low_pc
Lset6 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset6
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ## DW_AT_frame_base
	.byte	87
	.byte	5                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.long	64                              ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x34:0xb DW_TAG_variable
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.byte	7                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	4                               ## DW_AT_decl_line
	.long	68                              ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	4                               ## Abbrev [4] 0x40:0x4 DW_TAG_base_type
	.byte	6                               ## DW_AT_name
	.byte	5                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	5                               ## Abbrev [5] 0x44:0x10 DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.byte	9                               ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0x4a:0x9 DW_TAG_member
	.byte	8                               ## DW_AT_name
	.long	64                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	44                              ## Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang"                         ## string offset=0 ; clang
	.asciz	"main.cpp"                      ## string offset=6 ; main.cpp
	.asciz	"/^sdk"                         ## string offset=15 ; /^sdk
	.asciz	"MacOSX.sdk"                    ## string offset=21 ; MacOSX.sdk
	.asciz	"/^src"                         ## string offset=32 ; /^src
	.asciz	"main"                          ## string offset=38 ; main
	.asciz	"int"                           ## string offset=43 ; int
	.asciz	"f"                             ## string offset=47 ; f
	.asciz	"Foo"                           ## string offset=49 ; Foo
	.asciz	"x"                             ## string offset=53 ; x
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	6
	.long	15
	.long	21
	.long	32
	.long	38
	.long	43
	.long	47
	.long	53
	.long	49
	.section	__DWARF,__debug_addr,regular,debug
Lsection_info0:
Lset7 = Ldebug_addr_end0-Ldebug_addr_start0 ## Length of contribution
	.long	Lset7
Ldebug_addr_start0:
	.short	5                               ## DWARF version number
	.byte	8                               ## Address size
	.byte	0                               ## Segment selector size
Laddr_table_base0:
	.quad	Lfunc_begin0
Ldebug_addr_end0:
	.section	__DWARF,__debug_names,regular,debug
Ldebug_names_begin:
Lset8 = Lnames_end0-Lnames_start0       ## Header: unit length
	.long	Lset8
Lnames_start0:
	.short	5                               ## Header: version
	.short	0                               ## Header: padding
	.long	1                               ## Header: compilation unit count
	.long	0                               ## Header: local type unit count
	.long	0                               ## Header: foreign type unit count
	.long	3                               ## Header: bucket count
	.long	3                               ## Header: name count
Lset9 = Lnames_abbrev_end0-Lnames_abbrev_start0 ## Header: abbreviation table size
	.long	Lset9
	.long	8                               ## Header: augmentation string size
	.ascii	"LLVM0700"                      ## Header: augmentation string
Lset10 = Lcu_begin0-Lsection_info       ## Compilation unit 0
	.long	Lset10
	.long	1                               ## Bucket 0
	.long	2                               ## Bucket 1
	.long	3                               ## Bucket 2
	.long	193491849                       ## Hash in Bucket 0
	.long	2090499946                      ## Hash in Bucket 1
	.long	193495088                       ## Hash in Bucket 2
	.long	49                              ## String in Bucket 0: Foo
	.long	38                              ## String in Bucket 1: main
	.long	43                              ## String in Bucket 2: int
Lset11 = Lnames0-Lnames_entries0        ## Offset in Bucket 0
	.long	Lset11
Lset12 = Lnames1-Lnames_entries0        ## Offset in Bucket 1
	.long	Lset12
Lset13 = Lnames2-Lnames_entries0        ## Offset in Bucket 2
	.long	Lset13
Lnames_abbrev_start0:
	.byte	1                               ## Abbrev code
	.byte	19                              ## DW_TAG_structure_type
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	2                               ## Abbrev code
	.byte	46                              ## DW_TAG_subprogram
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	3                               ## Abbrev code
	.byte	36                              ## DW_TAG_base_type
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev list
Lnames_abbrev_end0:
Lnames_entries0:
Lnames0:
	.byte	1                               ## Abbreviation code
	.long	68                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: Foo
Lnames1:
	.byte	2                               ## Abbreviation code
	.long	37                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: main
Lnames2:
	.byte	3                               ## Abbreviation code
	.long	64                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: int
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
