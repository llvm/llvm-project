
; RUN: llc -debugger-tune=gdb -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s

source_filename = "2.c"
target datalayout = "E-m:a-p:32:32-i64:64-n32"

; Function Attrs: noinline nounwind optnone
define i32 @bar() #0 !dbg !8 {
entry:
  ret i32 1, !dbg !13
}

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 section "explicit_main_sec" !dbg !14 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 @bar(), !dbg !15
  ret i32 %call, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "2.c", directory: "debug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 13.0.0"}
!8 = distinct !DISubprogram(name: "bar", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "2.c", directory: "debug")
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 1, column: 12, scope: !8)
!14 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 2, type: !10, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 3, column: 10, scope: !14)
!16 = !DILocation(line: 3, column: 3, scope: !14)

; CHECK:               .csect .text[PR],5
; CHECK-NEXT:          .file   "2.c"
; CHECK-NEXT:          .globl  bar[DS]                         # -- Begin function bar
; CHECK-NEXT:          .globl  .bar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:          .csect bar[DS],2
; CHECK-NEXT:          .vbyte  4, .bar                         # @bar
; CHECK-NEXT:          .vbyte  4, TOC[TC0]
; CHECK-NEXT:          .vbyte  4, 0
; CHECK-NEXT:          .csect .text[PR],5
; CHECK-NEXT:  .bar:
; CHECK-NEXT:  L..func_begin0:
; CHECK-NEXT:  # %bb.0:                                # %entry
; CHECK-NEXT:  L..tmp0:
; CHECK-NEXT:  L..tmp1:
; CHECK-NEXT:          li 3, 1
; CHECK-NEXT:          blr
; CHECK-NEXT:  L..tmp2:
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
; CHECK-NEXT:          .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; CHECK-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-NEXT:          .vbyte  4, L..bar0-.bar                 # Function size
; CHECK-NEXT:          .vbyte  2, 0x0003                       # Function name len = 3
; CHECK-NEXT:          .byte   "bar"                           # Function Name
; CHECK-NEXT:  L..func_end0:
; CHECK-NEXT:                                          # -- End function
; CHECK-NEXT:          .csect explicit_main_sec[PR],5
; CHECK-NEXT:          .globl  main[DS]                        # -- Begin function main
; CHECK-NEXT:          .globl  .main
; CHECK-NEXT:          .align  2
; CHECK-NEXT:          .csect main[DS],2
; CHECK-NEXT:          .vbyte  4, .main                        # @main
; CHECK-NEXT:          .vbyte  4, TOC[TC0]
; CHECK-NEXT:          .vbyte  4, 0
; CHECK-NEXT:          .csect explicit_main_sec[PR],5
; CHECK-NEXT:  .main:
; CHECK-NEXT:  L..func_begin1:
; CHECK-NEXT:  # %bb.0:                                # %entry
; CHECK-NEXT:  L..tmp3:
; CHECK-NEXT:          mflr 0
; CHECK-NEXT:          stwu 1, -64(1)
; CHECK-NEXT:          li 3, 0
; CHECK-NEXT:          stw 0, 72(1)
; CHECK-NEXT:          stw 3, 60(1)
; CHECK-NEXT:  L..tmp4:
; CHECK-NEXT:  L..tmp5:
; CHECK-NEXT:          bl .bar
; CHECK-NEXT:          nop
; CHECK-NEXT:  L..tmp6:
; CHECK-NEXT:          addi 1, 1, 64
; CHECK-NEXT:          lwz 0, 8(1)
; CHECK-NEXT:          mtlr 0
; CHECK-NEXT:          blr
; CHECK-NEXT:  L..tmp7:
; CHECK-NEXT:  L..main0:
; CHECK-NEXT:          .vbyte  4, 0x00000000                   # Traceback table begin
; CHECK-NEXT:          .byte   0x00                            # Version = 0
; CHECK-NEXT:          .byte   0x09                            # Language = CPlusPlus
; CHECK-NEXT:          .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; CHECK-NEXT:                                          # +HasTraceBackTableOffset, -IsInternalProcedure
; CHECK-NEXT:                                          # -HasControlledStorage, -IsTOCless
; CHECK-NEXT:                                          # -IsFloatingPointPresent
; CHECK-NEXT:                                          # -IsFloatingPointOperationLogOrAbortEnabled
; CHECK-NEXT:          .byte   0x41                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; CHECK-NEXT:                                          # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
; CHECK-NEXT:          .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; CHECK-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; CHECK-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-NEXT:          .vbyte  4, L..main0-.main               # Function size
; CHECK-NEXT:          .vbyte  2, 0x0004                       # Function name len = 4
; CHECK-NEXT:          .byte   "main"                          # Function Name
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
; CHECK-NEXT:          .byte   1                               # Abbrev [1] 0xb:0x4f DW_TAG_compile_unit
; CHECK-NEXT:          .vbyte  4, L..info_string0              # DW_AT_producer
; CHECK-NEXT:          .vbyte  2, 12                           # DW_AT_language
; CHECK-NEXT:          .vbyte  4, L..info_string1              # DW_AT_name
; CHECK-NEXT:          .vbyte  4, L..line_table_start0         # DW_AT_stmt_list
; CHECK-NEXT:          .vbyte  4, L..info_string2              # DW_AT_comp_dir
; CHECK-NEXT:          .vbyte  4, 0                            # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..debug_ranges0             # DW_AT_ranges
; CHECK-NEXT:          .byte   2                               # Abbrev [2] 0x26:0x16 DW_TAG_subprogram
; CHECK-NEXT:          .vbyte  4, L..func_begin0               # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..func_end0                 # DW_AT_high_pc
; CHECK-NEXT:          .byte   1                               # DW_AT_frame_base
; CHECK-NEXT:          .byte   81
; CHECK-NEXT:          .vbyte  4, L..info_string3              # DW_AT_name
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_file
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_line
; CHECK-NEXT:          .vbyte  4, 82                           # DW_AT_type
; CHECK-NEXT:          .byte   1                               # DW_AT_external
; CHECK-NEXT:          .byte   2                               # Abbrev [2] 0x3c:0x16 DW_TAG_subprogram
; CHECK-NEXT:          .vbyte  4, L..func_begin1               # DW_AT_low_pc
; CHECK-NEXT:          .vbyte  4, L..func_end1                 # DW_AT_high_pc
; CHECK-NEXT:          .byte   1                               # DW_AT_frame_base
; CHECK-NEXT:          .byte   81
; CHECK-NEXT:          .vbyte  4, L..info_string5              # DW_AT_name
; CHECK-NEXT:          .byte   1                               # DW_AT_decl_file
; CHECK-NEXT:          .byte   2                               # DW_AT_decl_line
; CHECK-NEXT:          .vbyte  4, 82                           # DW_AT_type
; CHECK-NEXT:          .byte   1                               # DW_AT_external
; CHECK-NEXT:          .byte   3                               # Abbrev [3] 0x52:0x7 DW_TAG_base_type
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
; CHECK-NEXT:  	.string	"clang version 13.0.0"          # string offset=0
; CHECK-NEXT:  L..info_string1:
; CHECK-NEXT:  	.string	"2.c"                           # string offset=21
; CHECK-NEXT:  L..info_string2:
; CHECK-NEXT:  	.string	"debug"                         # string offset=25
; CHECK-NEXT:  L..info_string3:
; CHECK-NEXT:  	.string	"bar"                           # string offset=31
; CHECK-NEXT:  L..info_string4:
; CHECK-NEXT:  	.string	"int"                           # string offset=35
; CHECK-NEXT:  L..info_string5:
; CHECK-NEXT:  	.string	"main"                          # string offset=39
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
; CHECK-NEXT:          .byte   "debug"
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   "2.c"
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
; CHECK-NEXT:          .byte   1                               # Start sequence
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   12
; CHECK-NEXT:          .byte   10
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp1
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp1
; CHECK-NEXT:          .byte   3                               # Advance line 0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..func_end0
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..func_end0
; CHECK-NEXT:          .byte   0                               # End sequence
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp3
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp3
; CHECK-NEXT:          .byte   19                              # Start sequence
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   10
; CHECK-NEXT:          .byte   10
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp5
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp5
; CHECK-NEXT:          .byte   3                               # Advance line 1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   3
; CHECK-NEXT:          .byte   6
; CHECK-NEXT:          .byte   0                               # Set address to L..tmp6
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..tmp6
; CHECK-NEXT:          .byte   3                               # Advance line 0
; CHECK-NEXT:          .byte   0
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   0                               # Set address to L..func_end1
; CHECK-NEXT:          .byte   5
; CHECK-NEXT:          .byte   2
; CHECK-NEXT:          .vbyte  4, L..func_end1
; CHECK-NEXT:          .byte   0                               # End sequence
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:          .byte   1
; CHECK-NEXT:  L..debug_line_end0:
