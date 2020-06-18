##; Generated from the following manually stripped empty Swift program:
##;
##; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
##; target triple = "x86_64-apple-macosx10.9.0"
##;  
##; @__swift_reflection_version = linkonce_odr hidden constant i16 3
##; @llvm.used = appending global [1 x i8*] [i8* bitcast (i16* @__swift_reflection_version to i8*)], section "llvm.metadata", align 8
##;  
##; define i32 @main(i32, i8**) !dbg !29 {
##; entry:
##;   %2 = bitcast i8** %1 to i8*
##;   ret i32 0, !dbg !35
##; }
##;  
##; !llvm.dbg.cu = !{!0}
##; !swift.module.flags = !{!14}
##; !llvm.module.flags = !{!20, !21, !24}
##;  
##; !0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, enums: !2, imports: !3, sysroot: "/SDK")
##; !1 = !DIFile(filename: "ParseableInterfaceImports.swift", directory: "/")
##; !2 = !{}
##; !3 = !{!4, !6, !8}
##; !4 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !5, file: !1)
##; !5 = !DIModule(scope: null, name: "Foo", includePath: "/Foo/x86_64.swiftinterface")
##; !6 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !7, file: !1)
##; !7 = !DIModule(scope: null, name: "Swift", includePath: "/SDK/Swift.swiftmodule/x86_64.swiftinterface", sysroot: "/SDK")
##; !8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !7, file: !1)
##; !9 = !DIModule(scope: null, name: "Foundation", includePath: "/SDK/Foundation.swiftmodu
##; !14 = !{!"standard-library", i1 false}
##; !20 = !{i32 2, !"Dwarf Version", i32 4}
##; !21 = !{i32 2, !"Debug Info Version", i32 3}
##; !24 = !{i32 1, !"Swift Version", i32 7}
##; !29 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !5, file: !1, line: 1, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
##; !30 = !DISubroutineType(types: !31)
##; !31 = !{}
##; !35 = !DILocation(line: 0, scope: !36)
##; !36 = !DILexicalBlockFile(scope: !29, file: !37, discriminator: 0)
##; !37 = !DIFile(filename: "<compiler-generated>", directory: "")
        .section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 9
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
Lfunc_begin0:
	.file	1 "/ParseableInterfaceImports.swift"
	.loc	1 0 0                   ## ParseableInterfaceImports.swift:0:0
	.cfi_startproc
## %bb.0:                               ## %entry
	xorl	%eax, %eax
	retq
Ltmp0:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.private_extern	___swift_reflection_version ## @__swift_reflection_version
	.section	__TEXT,__const
	.globl	___swift_reflection_version
	.weak_definition	___swift_reflection_version
	.p2align	1
___swift_reflection_version:
	.short	3                       ## 0x3

	.no_dead_strip	___swift_reflection_version
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.byte	0                       ## string offset=0
	.asciz	"ParseableInterfaceImports.swift" ## string offset=1
	.asciz	"/"                     ## string offset=33
	.asciz	"Foo"                   ## string offset=35
	.asciz	"/Foo/x86_64.swiftinterface" ## string offset=39
	.asciz	"Swift"                 ## string offset=66
	.asciz	"/SDK/Swift.swiftmodule/x86_64.swiftinterface" ## string offset=72
	.asciz	"/SDK"                  ## string offset=117
	.asciz	"main"                  ## string offset=122
	.asciz	"Foundation"            ## string offset=127
        .asciz	"/SDK/Foundation.swiftmodule/x86_64.swiftinterface" ## string offset=138
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	14                      ## DW_FORM_strp
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.ascii	"\202|"                 ## DW_AT_LLVM_sysroot
	.byte	14                      ## DW_FORM_strp
	.byte	16                      ## DW_AT_stmt_list
	.byte	23                      ## DW_FORM_sec_offset
	.byte	27                      ## DW_AT_comp_dir
	.byte	14                      ## DW_FORM_strp
	.ascii	"\345\177"              ## DW_AT_APPLE_major_runtime_vers
	.byte	11                      ## DW_FORM_data1
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	30                      ## DW_TAG_module
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.ascii	"\200|"                 ## DW_AT_LLVM_include_path
	.byte	14                      ## DW_FORM_strp
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.ascii	"\347\177"              ## DW_AT_APPLE_omit_frame_ptr
	.byte	25                      ## DW_FORM_flag_present
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	110                     ## DW_AT_linkage_name
	.byte	14                      ## DW_FORM_strp
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	58                      ## DW_TAG_imported_module
	.byte	0                       ## DW_CHILDREN_no
	.byte	24                      ## DW_AT_import
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	5                       ## Abbreviation Code
	.byte	30                      ## DW_TAG_module
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.ascii	"\200|"                 ## DW_AT_LLVM_include_path
	.byte	14                      ## DW_FORM_strp
	.ascii	"\202|"                 ## DW_AT_LLVM_sysroot
	.byte	14                      ## DW_FORM_strp
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
 	.byte	30                      ## DW_TAG_module
	.byte	1                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.ascii	"\200|"                 ## DW_AT_LLVM_include_path
	.byte	14                      ## DW_FORM_strp
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                       ## DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0x5b DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	30                      ## DW_AT_language
	.long	1                       ## DW_AT_name
        .long	117                     ## DW_AT_name
.set Lset2, Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset2
	.long	33                      ## DW_AT_comp_dir
	.byte	5                       ## DW_AT_APPLE_major_runtime_vers
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset3
	.byte	2                       ## Abbrev [2] 0x2b:0x23 DW_TAG_module
	.long	35                      ## DW_AT_name
	.long	39                      ## DW_AT_LLVM_include_path
	.byte	3                       ## Abbrev [3] 0x34:0x19 DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset4, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset4
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                       ## DW_AT_frame_base
	.byte	87
	.long	122                     ## DW_AT_linkage_name
	.long	122                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
                                        ## DW_AT_external
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0x4e:0x5 DW_TAG_imported_module
	.long	47                      ## DW_AT_import
	.byte	5                       ## Abbrev [5] 0x53:0xd DW_TAG_module
	.long	66                      ## DW_AT_name
	.long	72                      ## DW_AT_LLVM_include_path
	.long	117                     ## DW_AT_LLVM_sysroot
	.byte	4                       ## Abbrev [4] 0x60:0x5 DW_TAG_imported_module
	.long	105                      ## DW_AT_import
	.byte	2                       ## Abbrev [2] 0x2b:0x23 DW_TAG_module
	.long	127                     ## DW_AT_name
	.long	138                     ## DW_AT_LLVM_include_path
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
Ldebug_info_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
