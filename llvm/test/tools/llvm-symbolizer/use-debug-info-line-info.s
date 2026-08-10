# REQUIRES: x86-registered-target
# Test llvm-symbolizer always uses line info from debug info if present.

# It's produced by the following steps.
# 1. Compile with "clang++ test.ll -S -o test.s".
# 2. Replace all "test.ll" with "<invalid>"" except the "test.ll" in first line.
# 3. Replace "/" in Linfo_string2 with "".
# source:
# ; ModuleID = 'test.ll'
# source_filename = "test.ll"
# ; Function Attrs: nounwind
# define void @foo(i32 %i) local_unnamed_addr #0 !dbg !5 {
# entry:
#     #dbg_value(i32 0, !9, !DIExpression(), !11)
#   switch i32 %i, label %if.end3 [
#     i32 5, label %if.end3.sink.split
#     i32 7, label %if.end3.sink.split
#   ], !dbg !11
# if.end3.sink.split:                               ; preds = %entry, %entry
#   tail call void @bar() #0, !dbg !12
#   br label %if.end3, !dbg !13
# if.end3:                                          ; preds = %if.end3.sink.split, %entry
#   tail call void @bar() #0, !dbg !13
#   ret void, !dbg !14
# }
# declare dso_local void @bar()
# attributes #0 = { nounwind }
# !llvm.dbg.cu = !{!0}
# !llvm.debugify = !{!2, !3}
# !llvm.module.flags = !{!4}
# !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
# !1 = !DIFile(filename: "test.ll", directory: "/")
# !2 = !{i32 7}
# !3 = !{i32 1}
# !4 = !{i32 2, !"Debug Info Version", i32 3}
# !5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
# !6 = !DISubroutineType(types: !7)
# !7 = !{}
# !8 = !{!9}
# !9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
# !10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
# !11 = !DILocation(line: 1, column: 1, scope: !5)
# !12 = !DILocation(line: 0, scope: !5)
# !13 = !DILocation(line: 6, column: 1, scope: !5)
# !14 = !DILocation(line: 7, column: 1, scope: !5)


# RUN: llvm-mc --filetype=obj --triple x86_64-pc-linux %s -o %t
# RUN: llvm-symbolizer --obj=%t 0xd | FileCheck %s
# RUN: llvm-symbolizer --inlining=false --obj=%t 0xd | FileCheck %s
# CHECK:      foo
# CHECK-NEXT: ??:0:0

	.file	"test.ll"
	.text
	.p2align	4
	.type	foo,@function
foo:                                    # @foo
.Lfunc_begin0:
	.file	1 "<invalid>"
	.loc	1 1 0                           # <invalid>:1:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	%edi, %eax
.Ltmp0:
	#DEBUG_VALUE: foo:1 <- 0
	.loc	1 1 1 prologue_end              # <invalid>:1:1
	orl	$2, %eax
	subl	$7, %eax
	je	.LBB0_1
	jmp	.LBB0_2
.Ltmp1:
.LBB0_1:                                # %if.end3.sink.split
	#DEBUG_VALUE: foo:1 <- 0
	.loc	1 0 0 is_stmt 0                 # <invalid>:0
	callq	bar
.Ltmp2:
.LBB0_2:                                # %if.end3
	#DEBUG_VALUE: foo:1 <- 0
	.loc	1 6 1 epilogue_begin is_stmt 1  # <invalid>:6:1
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	bar                             # TAILCALL
.Ltmp3:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x4d DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x26 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x43:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	80                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x50:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"debugify"                      # string offset=0
.Linfo_string1:
	.asciz	"<invalid>"                       # string offset=9
.Linfo_string2:
	.asciz	""                             # string offset=17
.Linfo_string3:
	.asciz	"foo"                           # string offset=19
.Linfo_string4:
	.asciz	"1"                             # string offset=23
.Linfo_string5:
	.asciz	"ty32"                          # string offset=25
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	88                              # Compilation Unit Length
	.long	42                              # DIE offset
	.asciz	"foo"                           # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	88                              # Compilation Unit Length
	.long	80                              # DIE offset
	.asciz	"ty32"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym bar
	.section	.debug_line,"",@progbits
.Lline_table_start0:
