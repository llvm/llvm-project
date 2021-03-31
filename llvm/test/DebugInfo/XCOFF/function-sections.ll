
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:   < %s | FileCheck %s

source_filename = "1.c"
target datalayout = "E-m:a-p:32:32-i64:64-n32"

; Function Attrs: noinline nounwind optnone
define i32 @foo() #0 !dbg !8 {
entry:
  ret i32 0, !dbg !12
}

; Function Attrs: noinline nounwind optnone
define i32 @bar() #0 !dbg !13 {
entry:
  ret i32 1, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "1.c", directory: "debug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 13.0.0"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)
!13 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 6, type: !9, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 8, column: 3, scope: !13)

; CHECK:               .csect .text[PR],2
; CHECK-NEXT:          .file   "1.c"
; CHECK-NEXT:          .csect .foo[PR],2
; CHECK-NEXT:          .globl  foo[DS]                         # -- Begin function foo
; CHECK-NEXT:          .globl  .foo[PR]
; CHECK-NEXT:          .align  2
; CHECK-NEXT:          .csect foo[DS],2
; CHECK-NEXT:          .vbyte  4, .foo[PR]                     # @foo
; CHECK-NEXT:          .vbyte  4, TOC[TC0]
; CHECK-NEXT:          .vbyte  4, 0
; CHECK-NEXT:          .csect .foo[PR],2
; CHECK-NEXT:  L..func_begin0:
; CHECK-NEXT:  # %bb.0:                                # %entry
; CHECK-NEXT:  L..tmp0:
; CHECK-NEXT:  L..tmp1:
; CHECK-NEXT:          li 3, 0
; CHECK-NEXT:          blr
; CHECK-NEXT:  L..tmp2:
; CHECK-NEXT:  L..foo0:
; CHECK-NEXT:          .vbyte  4, 0x00000000                   # Traceback table begin
; CHECK-NEXT:          .byte   0x00                            # Version = 0
; CHECK-NEXT:          .byte   0x09                            # Language = CPlusPlus
; CHECK-NEXT:          .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; CHECK-NEXT:                                          # +HasTraceBackTableOffset, -IsInternalProcedure
; CHECK-NEXT:                                          # -HasControlledStorage, -IsTOCless
; CHECK-NEXT:                                          # -IsFloatingPointPresent
; CHECK-NEXT:                                          # -IsFloatingPointOperationLogOrAbortEnabled
; CHECK-NEXT:          .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; CHECK-NEXT:                                          # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; CHECK-NEXT:          .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; CHECK-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-NEXT:          .vbyte  4, L..foo0-.foo[PR]             # Function size
; CHECK-NEXT:          .vbyte  2, 0x0003                       # Function name len = 3
; CHECK-NEXT:          .byte   'f,'o,'o                        # Function Name
; CHECK-NEXT:  L..func_end0:
; CHECK-NEXT:                                          # -- End function
; CHECK-NEXT:          .csect .bar[PR],2
; CHECK-NEXT:          .globl  bar[DS]                         # -- Begin function bar
; CHECK-NEXT:          .globl  .bar[PR]
; CHECK-NEXT:          .align  2
; CHECK-NEXT:          .csect bar[DS],2
; CHECK-NEXT:          .vbyte  4, .bar[PR]                     # @bar
; CHECK-NEXT:          .vbyte  4, TOC[TC0]
; CHECK-NEXT:          .vbyte  4, 0
; CHECK-NEXT:          .csect .bar[PR],2
; CHECK-NEXT:  L..func_begin1:
; CHECK-NEXT:  # %bb.0:                                # %entry
; CHECK-NEXT:  L..tmp3:
; CHECK-NEXT:  L..tmp4:
; CHECK-NEXT:          li 3, 1
; CHECK-NEXT:          blr
; CHECK-NEXT:  L..tmp5:
; CHECK-NEXT:  L..bar0:
; CHECK-NEXT:          .vbyte  4, 0x00000000                   # Traceback table begin
; CHECK-NEXT:          .byte   0x00                            # Version = 0
; CHECK-NEXT:          .byte   0x09                            # Language = CPlusPlus
; CHECK-NEXT:          .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; CHECK-NEXT:                                          # +HasTraceBackTableOffset, -IsInternalProcedure
; CHECK-NEXT:                                          # -HasControlledStorage, -IsTOCless
; CHECK-NEXT:                                          # -IsFloatingPointPresent
; CHECK-NEXT:                                          # -IsFloatingPointOperationLogOrAbortEnabled
; CHECK-NEXT:          .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; CHECK-NEXT:                                          # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; CHECK-NEXT:          .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; CHECK-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-NEXT:          .vbyte  4, L..bar0-.bar[PR]             # Function size
; CHECK-NEXT:          .vbyte  2, 0x0003                       # Function name len = 3
; CHECK-NEXT:          .byte   'b,'a,'r                        # Function Name
; CHECK-NEXT:  L..func_end1:
; CHECK-NEXT:                                          # -- End function
; CHECK-NEXT:  L..sec_end0:
; CHECK:               .dwsect 0x60000
; CHECK-NEXT:  L...dwabrev:
; CHECK-NEXT:          .byte   1                               # Abbreviation Code
; CHECK-NEXT:          .byte   17                              # DW_TAG_compile_unit
; CHECK-NEXT:          .byte   1                               # DW_CHILDREN_yes
; CHECK-NEXT:          .byte   37                              # DW_AT_producer
; CHECK-NEXT:          .byte   14                              # DW_FORM_strp
; CHECK-NEXT:          .byte   19                              # DW_AT_language
; CHECK-NEXT:          .byte   5                               # DW_FORM_data2
; CHECK-NEXT:          .byte   3                               # DW_AT_name
; CHECK-NEXT:          .byte   14                              # DW_FORM_strp
; CHECK-NEXT:          .byte   16                              # DW_AT_stmt_list
; CHECK-NEXT:          .byte   6                               # DW_FORM_data4
; CHECK-NEXT:          .byte   27                              # DW_AT_comp_dir
; CHECK-NEXT:          .byte   14                              # DW_FORM_strp
; CHECK-NEXT:          .byte   17                              # DW_AT_low_pc
; CHECK-NEXT:          .byte   1                               # DW_FORM_addr
; CHECK-NEXT:          .byte   85                              # DW_AT_ranges
; CHECK-NEXT:          .byte   6                               # DW_FORM_data4
; CHECK-NEXT:          .byte   0                               # EOM(1)
; CHECK-NEXT:          .byte   0                               # EOM(2)
; CHECK-NEXT:          .byte   2                               # Abbreviation Code
; CHECK-NEXT:          .byte   46                              # DW_TAG_subprogram
; CHECK-NEXT:          .byte   0                               # DW_CHILDREN_no
; CHECK-NEXT:          .byte   17                              # DW_AT_low_pc
; CHECK-NEXT:          .byte   1                               # DW_FORM_addr
; CHECK-NEXT:          .byte   18                              # DW_AT_high_pc
; CHECK-NEXT:          .byte   1                               # DW_FORM_addr
; CHECK-NEXT:          .byte   64                              # DW_AT_frame_base
; CHECK-NEXT:          .byte   10                              # DW_FORM_block1
; CHECK-NEXT:          .byte   3                               # DW_AT_name
; CHECK-NEXT:          .byte   14                              # DW_FORM_strp
; CHECK-NEXT:          .byte   58                              # DW_AT_decl_file
; CHECK-NEXT:          .byte   11                              # DW_FORM_data1
; CHECK-NEXT:          .byte   59                              # DW_AT_decl_line
; CHECK-NEXT:          .byte   11                              # DW_FORM_data1
; CHECK-NEXT:          .byte   39                              # DW_AT_prototyped
; CHECK-NEXT:          .byte   12                              # DW_FORM_flag
; CHECK-NEXT:          .byte   73                              # DW_AT_type
; CHECK-NEXT:          .byte   19                              # DW_FORM_ref4
; CHECK-NEXT:          .byte   63                              # DW_AT_external
; CHECK-NEXT:          .byte   12                              # DW_FORM_flag
; CHECK-NEXT:          .byte   0                               # EOM(1)
; CHECK-NEXT:          .byte   0                               # EOM(2)
; CHECK-NEXT:          .byte   3                               # Abbreviation Code
; CHECK-NEXT:          .byte   36                              # DW_TAG_base_type
; CHECK-NEXT:          .byte   0                               # DW_CHILDREN_no
; CHECK-NEXT:          .byte   3                               # DW_AT_name
; CHECK-NEXT:          .byte   14                              # DW_FORM_strp
; CHECK-NEXT:          .byte   62                              # DW_AT_encoding
; CHECK-NEXT:          .byte   11                              # DW_FORM_data1
; CHECK-NEXT:          .byte   11                              # DW_AT_byte_size
; CHECK-NEXT:          .byte   11                              # DW_FORM_data1
; CHECK-NEXT:          .byte   0                               # EOM(1)
; CHECK-NEXT:          .byte   0                               # EOM(2)
; CHECK-NEXT:          .byte   0                               # EOM(3)
; CHECK:               .dwsect 0x10000
; CHECK-NEXT:  L...dwinfo:
; CHECK-NEXT:  L..cu_begin0:
; CHECK-NEXT:          .vbyte  2, 3                            # DWARF version number
; CHECK-NEXT:          .vbyte  4, L...dwabrev                  # Offset Into Abbrev. Section
; CHECK-NEXT:          .byte   4                               # Address Size (in bytes)
; CHECK-NEXT:          .byte   1                               # Abbrev [1] 0xb:0x51 DW_TAG_compile_unit
; CHECK-NEXT:          .vbyte  4, L..info_string0              # DW_AT_producer
; CHECK-NEXT:          .vbyte  2, 12                           # DW_AT_language
; CHECK-NEXT:          .vbyte  4, L..info_string1              # DW_AT_name
; CHECK-NEXT:          .vbyte  4, L..line_table_start0         # DW_AT_stmt_list
; CHECK-NEXT:          .vbyte  4, L..info_string2              # DW_AT_comp_dir
; CHECK-NEXT:          .vbyte  4, 0                            # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..debug_ranges0             # DW_AT_ranges
; CHECK-NEXT:          .byte   2                               # Abbrev [2] 0x26:0x17 DW_TAG_subprogram
; CHECK-NEXT:          .vbyte  4, L..func_begin0               # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..func_end0                 # DW_AT_high_pc
; CHECK-NEXT:          .byte   1                               # DW_AT_frame_base
; CHECK-NEXT:          .byte   81
; CHECK-NEXT:          .vbyte  4, L..info_string3              # DW_AT_name
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_file
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_line
; CHECK-NEXT:          .byte   1                               # DW_AT_prototyped
; CHECK-NEXT:          .vbyte  4, 84                           # DW_AT_type
; CHECK-NEXT:          .byte   1                               # DW_AT_external
; CHECK-NEXT:          .byte   2                               # Abbrev [2] 0x3d:0x17 DW_TAG_subprogram
; CHECK-NEXT:          .vbyte  4, L..func_begin1               # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..func_end1                 # DW_AT_high_pc
; CHECK-NEXT:          .byte   1                               # DW_AT_frame_base
; CHECK-NEXT:          .byte   81
; CHECK-NEXT:          .vbyte  4, L..info_string5              # DW_AT_name
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_file
; CHECK-NEXT:          .byte   6                               # DW_AT_decl_line
; CHECK-NEXT:          .byte   1                               # DW_AT_prototyped
; CHECK-NEXT:          .vbyte  4, 84                           # DW_AT_type
; CHECK-NEXT:          .byte   1                               # DW_AT_external
; CHECK-NEXT:          .byte   3                               # Abbrev [3] 0x54:0x7 DW_TAG_base_type
; CHECK-NEXT:          .vbyte  4, L..info_string4              # DW_AT_name
; CHECK-NEXT:          .byte   5                               # DW_AT_encoding
; CHECK-NEXT:          .byte   4                               # DW_AT_byte_size
; CHECK-NEXT:          .byte   0                               # End Of Children Mark
; CHECK-NEXT:  L..debug_info_end0:
; CHECK:               .dwsect 0x80000
; CHECK-NEXT:  L...dwrnges:
; CHECK-NEXT:  L..debug_ranges0:
; CHECK-NEXT:          .vbyte  4, L..func_begin0
; CHECK-NEXT:          .vbyte  4, L..func_end0
; CHECK-NEXT:          .vbyte  4, L..func_begin1
; CHECK-NEXT:          .vbyte  4, L..func_end1
; CHECK-NEXT:          .vbyte  4, 0
; CHECK-NEXT:          .vbyte  4, 0
; CHECK:               .dwsect 0x70000
; CHECK-NEXT:  L...dwstr:
; CHECK-NEXT:  L..info_string0:
; CHECK-NEXT:          .byte   'c,'l,'a,'n,'g,' ,'v,'e,'r,'s,'i,'o,'n,' ,'1,'3,'.,'0,'.,'0,0000 # string offset=0
; CHECK-NEXT:  L..info_string1:
; CHECK-NEXT:          .byte   '1,'.,'c,0000                   # string offset=21
; CHECK-NEXT:  L..info_string2:
; CHECK-NEXT:          .byte   'd,'e,'b,'u,'g,0000             # string offset=25
; CHECK-NEXT:  L..info_string3:
; CHECK-NEXT:          .byte   'f,'o,'o,0000                   # string offset=31
; CHECK-NEXT:  L..info_string4:
; CHECK-NEXT:          .byte   'i,'n,'t,0000                   # string offset=35
; CHECK-NEXT:  L..info_string5:
; CHECK-NEXT:          .byte   'b,'a,'r,0000                   # string offset=39
; CHECK-NEXT:          .toc
; CHECK:               .dwsect 0x20000
; CHECK-NEXT:  L...dwline:
; CHECK-NEXT:  L..debug_line_0:
; CHECK-NEXT:  .set L..line_table_start0, L..debug_line_0-4
; CHECK-NEXT:          .vbyte  2, 3
; CHECK-NEXT:          .vbyte  4, L..prologue_end0-L..prologue_start0
; CHECK-NEXT:  L..prologue_start0:
; CHECK-NEXT:          .byte   4
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   -5
; CHECK-NEXT:          .byte   14
; CHECK-NEXT:          .byte   13
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   'd,'e,'b,'u,'g
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   '1,'.,'c
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:  L..prologue_end0:
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp0
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp0
; CHECK-NEXT:          .byte   19                              # Start sequence
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   3
; CHECK-NEXT:          .byte   10
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp1
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp1
; CHECK-NEXT:          .byte   3                               # Advance line 1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..sec_end0
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..sec_end0
; CHECK-NEXT:          .byte   0                               # End sequence
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp3
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp3
; CHECK-NEXT:          .byte   24                              # Start sequence
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   3
; CHECK-NEXT:          .byte   10
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp4
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp4
; CHECK-NEXT:          .byte   3                               # Advance line 1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..sec_end0
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..sec_end0
; CHECK-NEXT:          .byte   0                               # End sequence
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:  L..debug_line_end0:
