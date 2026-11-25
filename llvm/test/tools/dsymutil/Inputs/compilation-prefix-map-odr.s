## Two C++ files compiled with prefix mapping that define different types with
## the same name on the same line of headers with the same name. Each part is
## generated with llc -O0 from the IR in its comments.

#--- a.s
##; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
##; target triple = "x86_64-apple-macosx10.15.0"
##;
##; %struct.Foo = type { i32 }
##;
##; define i32 @main() !dbg !10 {
##; entry:
##;   %a = alloca %struct.Foo, align 4
##;     #dbg_declare(ptr %a, !14, !DIExpression(), !15)
##;   ret i32 0, !dbg !15
##; }
##;
##; !llvm.dbg.cu = !{!0}
##; !llvm.module.flags = !{!40, !41}
##;
##; !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
##; !1 = !DIFile(filename: "a.cpp", directory: "/^src/a")
##; !2 = !DIFile(filename: "foo.h", directory: "/^src/a")
##; !3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !2, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !4)
##; !4 = !{!5}
##; !5 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !3, file: !2, line: 1, baseType: !6, size: 32)
##; !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
##; !10 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
##; !11 = !DISubroutineType(types: !12)
##; !12 = !{!6}
##; !14 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 4, type: !3)
##; !15 = !DILocation(line: 4, column: 7, scope: !10)
##; !40 = !{i32 7, !"Dwarf Version", i32 5}
##; !41 = !{i32 2, !"Debug Info Version", i32 3}

	.build_version macos, 10, 15
	.section	__TEXT,__text,regular,pure_instructions
	.att_syntax
	.globl	_main                           ## -- Begin function main
	.p2align	4
_main:                                  ## @main
Lfunc_begin0:
	.file	0 "/^src/a" "a.cpp"
	.cfi_startproc
## %bb.0:                               ## %entry
	.loc	0 4 7 prologue_end              ## a.cpp:4:7
	xorl	%eax, %eax
	retq
Ltmp0:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.file	1 "foo.h"
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
	.byte	1                               ## Abbrev [1] 0xc:0x47 DW_TAG_compile_unit
	.byte	0                               ## DW_AT_producer
	.short	33                              ## DW_AT_language
	.byte	1                               ## DW_AT_name
Lset2 = Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
	.long	Lset2
Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset3
	.byte	2                               ## DW_AT_comp_dir
	.byte	0                               ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
Lset5 = Laddr_table_base0-Lsection_info0 ## DW_AT_addr_base
	.long	Lset5
	.byte	2                               ## Abbrev [2] 0x23:0x1b DW_TAG_subprogram
	.byte	0                               ## DW_AT_low_pc
Lset6 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset6
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ## DW_AT_frame_base
	.byte	87
	.byte	3                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.long	62                              ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x32:0xb DW_TAG_variable
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.byte	5                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	4                               ## DW_AT_decl_line
	.long	66                              ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	4                               ## Abbrev [4] 0x3e:0x4 DW_TAG_base_type
	.byte	4                               ## DW_AT_name
	.byte	5                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	5                               ## Abbrev [5] 0x42:0x10 DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.byte	7                               ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0x48:0x9 DW_TAG_member
	.byte	6                               ## DW_AT_name
	.long	62                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	36                              ## Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang"                         ## string offset=0 ; clang
	.asciz	"a.cpp"                         ## string offset=6 ; a.cpp
	.asciz	"/^src/a"                       ## string offset=12 ; /^src/a
	.asciz	"main"                          ## string offset=20 ; main
	.asciz	"int"                           ## string offset=25 ; int
	.asciz	"a"                             ## string offset=29 ; a
	.asciz	"Foo"                           ## string offset=31 ; Foo
	.asciz	"x"                             ## string offset=35 ; x
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	6
	.long	12
	.long	20
	.long	25
	.long	29
	.long	35
	.long	31
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
	.long	31                              ## String in Bucket 0: Foo
	.long	20                              ## String in Bucket 1: main
	.long	25                              ## String in Bucket 2: int
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
	.long	66                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: Foo
Lnames1:
	.byte	2                               ## Abbreviation code
	.long	35                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: main
Lnames2:
	.byte	3                               ## Abbreviation code
	.long	62                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: int
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:

#--- b.s
##; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
##; target triple = "x86_64-apple-macosx10.15.0"
##;
##; %struct.Foo = type { i32 }
##;
##; define i32 @f() !dbg !10 {
##; entry:
##;   %b = alloca %struct.Foo, align 4
##;     #dbg_declare(ptr %b, !14, !DIExpression(), !15)
##;   ret i32 0, !dbg !15
##; }
##;
##; !llvm.dbg.cu = !{!0}
##; !llvm.module.flags = !{!40, !41}
##;
##; !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
##; !1 = !DIFile(filename: "b.cpp", directory: "/^src/b")
##; !2 = !DIFile(filename: "foo.h", directory: "/^src/b")
##; !3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !2, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !4)
##; !4 = !{!5}
##; !5 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !3, file: !2, line: 1, baseType: !6, size: 32)
##; !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
##; !10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
##; !11 = !DISubroutineType(types: !12)
##; !12 = !{!6}
##; !14 = !DILocalVariable(name: "b", scope: !10, file: !1, line: 4, type: !3)
##; !15 = !DILocation(line: 4, column: 7, scope: !10)
##; !40 = !{i32 7, !"Dwarf Version", i32 5}
##; !41 = !{i32 2, !"Debug Info Version", i32 3}

	.build_version macos, 10, 15
	.section	__TEXT,__text,regular,pure_instructions
	.att_syntax
	.globl	_f                              ## -- Begin function f
	.p2align	4
_f:                                     ## @f
Lfunc_begin0:
	.file	0 "/^src/b" "b.cpp"
	.cfi_startproc
## %bb.0:                               ## %entry
	.loc	0 4 7 prologue_end              ## b.cpp:4:7
	xorl	%eax, %eax
	retq
Ltmp0:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.file	1 "foo.h"
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
	.byte	1                               ## Abbrev [1] 0xc:0x47 DW_TAG_compile_unit
	.byte	0                               ## DW_AT_producer
	.short	33                              ## DW_AT_language
	.byte	1                               ## DW_AT_name
Lset2 = Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
	.long	Lset2
Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset3
	.byte	2                               ## DW_AT_comp_dir
	.byte	0                               ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
Lset5 = Laddr_table_base0-Lsection_info0 ## DW_AT_addr_base
	.long	Lset5
	.byte	2                               ## Abbrev [2] 0x23:0x1b DW_TAG_subprogram
	.byte	0                               ## DW_AT_low_pc
Lset6 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset6
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ## DW_AT_frame_base
	.byte	87
	.byte	3                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.long	62                              ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x32:0xb DW_TAG_variable
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.byte	5                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	4                               ## DW_AT_decl_line
	.long	66                              ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	4                               ## Abbrev [4] 0x3e:0x4 DW_TAG_base_type
	.byte	4                               ## DW_AT_name
	.byte	5                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	5                               ## Abbrev [5] 0x42:0x10 DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.byte	7                               ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0x48:0x9 DW_TAG_member
	.byte	6                               ## DW_AT_name
	.long	62                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	36                              ## Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang"                         ## string offset=0 ; clang
	.asciz	"b.cpp"                         ## string offset=6 ; b.cpp
	.asciz	"/^src/b"                       ## string offset=12 ; /^src/b
	.asciz	"f"                             ## string offset=20 ; f
	.asciz	"int"                           ## string offset=22 ; int
	.asciz	"b"                             ## string offset=26 ; b
	.asciz	"Foo"                           ## string offset=28 ; Foo
	.asciz	"y"                             ## string offset=32 ; y
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	6
	.long	12
	.long	20
	.long	22
	.long	26
	.long	32
	.long	28
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
	.long	0                               ## Bucket 1
	.long	3                               ## Bucket 2
	.long	177675                          ## Hash in Bucket 0
	.long	193491849                       ## Hash in Bucket 0
	.long	193495088                       ## Hash in Bucket 2
	.long	20                              ## String in Bucket 0: f
	.long	28                              ## String in Bucket 0: Foo
	.long	22                              ## String in Bucket 2: int
Lset11 = Lnames0-Lnames_entries0        ## Offset in Bucket 0
	.long	Lset11
Lset12 = Lnames1-Lnames_entries0        ## Offset in Bucket 0
	.long	Lset12
Lset13 = Lnames2-Lnames_entries0        ## Offset in Bucket 2
	.long	Lset13
Lnames_abbrev_start0:
	.byte	1                               ## Abbrev code
	.byte	46                              ## DW_TAG_subprogram
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	2                               ## Abbrev code
	.byte	19                              ## DW_TAG_structure_type
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
	.long	35                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: f
Lnames1:
	.byte	2                               ## Abbreviation code
	.long	66                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: Foo
Lnames2:
	.byte	3                               ## Abbreviation code
	.long	62                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: int
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
