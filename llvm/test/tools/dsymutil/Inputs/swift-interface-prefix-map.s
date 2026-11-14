##; Generated with llc -O0 from the following IR, which mimics an empty Swift
##; program compiled with prefix mapping:
##;
##; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
##; target triple = "x86_64-apple-macosx10.9.0"
##;
##; define i32 @main(i32 %0, ptr %1) !dbg !20 {
##; entry:
##;   ret i32 0, !dbg !23
##; }
##;
##; !llvm.dbg.cu = !{!0}
##; !llvm.module.flags = !{!30, !31}
##;
##; !0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, imports: !3, sysroot: "/^sdk")
##; !1 = !DIFile(filename: "main.swift", directory: "/^src")
##; !2 = !{}
##; !3 = !{!4, !6, !8, !10, !12}
##; !4 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !5, file: !1)
##; !5 = !DIModule(scope: null, name: "Foo", includePath: "/^src/Foo/x86_64.swiftinterface")
##; !6 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !7, file: !1)
##; !7 = !DIModule(scope: null, name: "Bar", includePath: "Bar/x86_64.swiftinterface")
##; !8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !9, file: !1)
##; !9 = !DIModule(scope: null, name: "Swift", includePath: "/^sdk/usr/lib/swift/Swift.swiftmodule/x86_64-apple-macos.swiftinterface")
##; !10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !11, file: !1)
##; !11 = !DIModule(scope: null, name: "Testing", includePath: "/^xcode/Platforms/MacOSX.platform/Developer/Library/Frameworks/Testing.framework/Modules/Testing.swiftmodule/x86_64-apple-macos.swiftinterface")
##; !12 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !13, file: !1)
##; !13 = !DIModule(scope: null, name: "_Concurrency", includePath: "/^toolchain/usr/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftinterface")
##; !20 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 1, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
##; !21 = !DISubroutineType(types: !2)
##; !23 = !DILocation(line: 1, scope: !20)
##; !30 = !{i32 2, !"Dwarf Version", i32 4}
##; !31 = !{i32 2, !"Debug Info Version", i32 3}

	.macosx_version_min 10, 9
	.section	__TEXT,__text,regular,pure_instructions
	.att_syntax
	.globl	_main                           ## -- Begin function main
	.p2align	4
_main:                                  ## @main
Lfunc_begin0:
	.cfi_startproc
## %bb.0:                               ## %entry
	.file	1 "/^src" "main.swift"
	.loc	1 1 0 prologue_end              ## main.swift:1
	xorl	%eax, %eax
	retq
Ltmp0:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ## Abbreviation Code
	.byte	17                              ## DW_TAG_compile_unit
	.byte	1                               ## DW_CHILDREN_yes
	.byte	37                              ## DW_AT_producer
	.byte	14                              ## DW_FORM_strp
	.byte	19                              ## DW_AT_language
	.byte	5                               ## DW_FORM_data2
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.ascii	"\202|"                         ## DW_AT_LLVM_sysroot
	.byte	14                              ## DW_FORM_strp
	.byte	16                              ## DW_AT_stmt_list
	.byte	23                              ## DW_FORM_sec_offset
	.byte	27                              ## DW_AT_comp_dir
	.byte	14                              ## DW_FORM_strp
	.ascii	"\345\177"                      ## DW_AT_APPLE_major_runtime_vers
	.byte	11                              ## DW_FORM_data1
	.byte	17                              ## DW_AT_low_pc
	.byte	1                               ## DW_FORM_addr
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	2                               ## Abbreviation Code
	.byte	46                              ## DW_TAG_subprogram
	.byte	0                               ## DW_CHILDREN_no
	.byte	17                              ## DW_AT_low_pc
	.byte	1                               ## DW_FORM_addr
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.ascii	"\347\177"                      ## DW_AT_APPLE_omit_frame_ptr
	.byte	25                              ## DW_FORM_flag_present
	.byte	64                              ## DW_AT_frame_base
	.byte	24                              ## DW_FORM_exprloc
	.byte	110                             ## DW_AT_linkage_name
	.byte	14                              ## DW_FORM_strp
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	3                               ## Abbreviation Code
	.byte	30                              ## DW_TAG_module
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.ascii	"\200|"                         ## DW_AT_LLVM_include_path
	.byte	14                              ## DW_FORM_strp
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	4                               ## Abbreviation Code
	.byte	58                              ## DW_TAG_imported_module
	.byte	0                               ## DW_CHILDREN_no
	.byte	24                              ## DW_AT_import
	.byte	19                              ## DW_FORM_ref4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	0                               ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                               ## DWARF version number
Lset1 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                               ## Address Size (in bytes)
	.byte	1                               ## Abbrev [1] 0xb:0x84 DW_TAG_compile_unit
	.long	0                               ## DW_AT_producer
	.short	30                              ## DW_AT_language
	.long	1                               ## DW_AT_name
	.long	12                              ## DW_AT_LLVM_sysroot
Lset2 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset2
	.long	18                              ## DW_AT_comp_dir
	.byte	5                               ## DW_AT_APPLE_major_runtime_vers
	.quad	Lfunc_begin0                    ## DW_AT_low_pc
Lset3 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset3
	.byte	2                               ## Abbrev [2] 0x2f:0x19 DW_TAG_subprogram
	.quad	Lfunc_begin0                    ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                               ## DW_AT_frame_base
	.byte	87
	.long	24                              ## DW_AT_linkage_name
	.long	24                              ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x48:0x9 DW_TAG_module
	.long	29                              ## DW_AT_name
	.long	33                              ## DW_AT_LLVM_include_path
	.byte	4                               ## Abbrev [4] 0x51:0x5 DW_TAG_imported_module
	.long	72                              ## DW_AT_import
	.byte	3                               ## Abbrev [3] 0x56:0x9 DW_TAG_module
	.long	65                              ## DW_AT_name
	.long	69                              ## DW_AT_LLVM_include_path
	.byte	4                               ## Abbrev [4] 0x5f:0x5 DW_TAG_imported_module
	.long	86                              ## DW_AT_import
	.byte	3                               ## Abbrev [3] 0x64:0x9 DW_TAG_module
	.long	95                              ## DW_AT_name
	.long	101                             ## DW_AT_LLVM_include_path
	.byte	4                               ## Abbrev [4] 0x6d:0x5 DW_TAG_imported_module
	.long	100                             ## DW_AT_import
	.byte	3                               ## Abbrev [3] 0x72:0x9 DW_TAG_module
	.long	173                             ## DW_AT_name
	.long	181                             ## DW_AT_LLVM_include_path
	.byte	4                               ## Abbrev [4] 0x7b:0x5 DW_TAG_imported_module
	.long	114                             ## DW_AT_import
	.byte	3                               ## Abbrev [3] 0x80:0x9 DW_TAG_module
	.long	324                             ## DW_AT_name
	.long	337                             ## DW_AT_LLVM_include_path
	.byte	4                               ## Abbrev [4] 0x89:0x5 DW_TAG_imported_module
	.long	128                             ## DW_AT_import
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.byte	0                               ## string offset=0 ; 
	.asciz	"main.swift"                    ## string offset=1 ; main.swift
	.asciz	"/^sdk"                         ## string offset=12 ; /^sdk
	.asciz	"/^src"                         ## string offset=18 ; /^src
	.asciz	"main"                          ## string offset=24 ; main
	.asciz	"Foo"                           ## string offset=29 ; Foo
	.asciz	"/^src/Foo/x86_64.swiftinterface" ## string offset=33 ; /^src/Foo/x86_64.swiftinterface
	.asciz	"Bar"                           ## string offset=65 ; Bar
	.asciz	"Bar/x86_64.swiftinterface"     ## string offset=69 ; Bar/x86_64.swiftinterface
	.asciz	"Swift"                         ## string offset=95 ; Swift
	.asciz	"/^sdk/usr/lib/swift/Swift.swiftmodule/x86_64-apple-macos.swiftinterface" ## string offset=101 ; /^sdk/usr/lib/swift/Swift.swiftmodule/x86_64-apple-macos.swiftinterface
	.asciz	"Testing"                       ## string offset=173 ; Testing
	.asciz	"/^xcode/Platforms/MacOSX.platform/Developer/Library/Frameworks/Testing.framework/Modules/Testing.swiftmodule/x86_64-apple-macos.swiftinterface" ## string offset=181 ; /^xcode/Platforms/MacOSX.platform/Developer/Library/Frameworks/Testing.framework/Modules/Testing.swiftmodule/x86_64-apple-macos.swiftinterface
	.asciz	"_Concurrency"                  ## string offset=324 ; _Concurrency
	.asciz	"/^toolchain/usr/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftinterface" ## string offset=337 ; /^toolchain/usr/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftinterface
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	1                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	0                               ## Bucket 0
	.long	2090499946                      ## Hash in Bucket 0
Lset5 = LNames0-Lnames_begin            ## Offset in Bucket 0
	.long	Lset5
LNames0:
	.long	24                              ## main
	.long	1                               ## Num DIEs
	.long	47
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	0                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	-1                              ## Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	0                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	-1                              ## Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	0                               ## Header Hash Count
	.long	20                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	3                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.short	3                               ## DW_ATOM_die_tag
	.short	5                               ## DW_FORM_data2
	.short	4                               ## DW_ATOM_type_flags
	.short	11                              ## DW_FORM_data1
	.long	-1                              ## Bucket 0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
