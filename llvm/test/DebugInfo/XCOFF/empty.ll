
; RUN: llc -debugger-tune=gdb -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=ASM32
; RUN: llc -debugger-tune=gdb -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=ASM64

source_filename = "1.c"
target datalayout = "E-m:a-p:32:32-i64:64-n32"

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "1.c", directory: "debug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 12.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)

; ASM32:               .csect .text[PR],2
; ASM32-NEXT:          .file   "1.c"
; ASM32-NEXT:          .globl  main[DS]                        # -- Begin function main
; ASM32-NEXT:          .globl  .main
; ASM32-NEXT:          .align  2
; ASM32-NEXT:          .csect main[DS],2
; ASM32-NEXT:          .vbyte  4, .main                        # @main
; ASM32-NEXT:          .vbyte  4, TOC[TC0]
; ASM32-NEXT:          .vbyte  4, 0
; ASM32-NEXT:          .csect .text[PR],2
; ASM32-NEXT:  .main:
; ASM32-NEXT:  L..func_begin0:
; ASM32-NEXT:  # %bb.0:                                # %entry
; ASM32-NEXT:  L..tmp0:
; ASM32-NEXT:          li 4, 0
; ASM32-NEXT:  L..tmp1:
; ASM32-NEXT:  L..tmp2:
; ASM32-NEXT:          li 3, 0
; ASM32-NEXT:          stw 4, -4(1)
; ASM32-NEXT:          blr
; ASM32-NEXT:  L..tmp3:
; ASM32-NEXT:  L..main0:
; ASM32-NEXT:          .vbyte  4, 0x00000000                   # Traceback table begin
; ASM32-NEXT:          .byte   0x00                            # Version = 0
; ASM32-NEXT:          .byte   0x09                            # Language = CPlusPlus
; ASM32-NEXT:          .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; ASM32-NEXT:                                          # +HasTraceBackTableOffset, -IsInternalProcedure
; ASM32-NEXT:                                          # -HasControlledStorage, -IsTOCless
; ASM32-NEXT:                                          # -IsFloatingPointPresent
; ASM32-NEXT:                                          # -IsFloatingPointOperationLogOrAbortEnabled
; ASM32-NEXT:          .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; ASM32-NEXT:                                          # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; ASM32-NEXT:          .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; ASM32-NEXT:          .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; ASM32-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; ASM32-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; ASM32-NEXT:          .vbyte  4, L..main0-.main               # Function size
; ASM32-NEXT:          .vbyte  2, 0x0004                       # Function name len = 4
; ASM32-NEXT:          .byte   "main"                          # Function Name
; ASM32-NEXT:  L..func_end0:
; ASM32-NEXT:                                          # -- End function
; ASM32-NEXT:  L..sec_end0:
; ASM32:               .dwsect 0x60000
; ASM32-NEXT:  L...dwabrev:
; ASM32-NEXT:          .byte   1                               # Abbreviation Code
; ASM32-NEXT:          .byte   17                              # DW_TAG_compile_unit
; ASM32-NEXT:          .byte   1                               # DW_CHILDREN_yes
; ASM32-NEXT:          .byte   37                              # DW_AT_producer
; ASM32-NEXT:          .byte   14                              # DW_FORM_strp
; ASM32-NEXT:          .byte   19                              # DW_AT_language
; ASM32-NEXT:          .byte   5                               # DW_FORM_data2
; ASM32-NEXT:          .byte   3                               # DW_AT_name
; ASM32-NEXT:          .byte   14                              # DW_FORM_strp
; ASM32-NEXT:          .byte   16                              # DW_AT_stmt_list
; ASM32-NEXT:          .byte   23                              # DW_FORM_sec_offset
; ASM32-NEXT:          .byte   27                              # DW_AT_comp_dir
; ASM32-NEXT:          .byte   14                              # DW_FORM_strp
; ASM32-NEXT:          .byte   17                              # DW_AT_low_pc
; ASM32-NEXT:          .byte   1                               # DW_FORM_addr
; ASM32-NEXT:          .byte   18                              # DW_AT_high_pc
; ASM32-NEXT:          .byte   6                               # DW_FORM_data4
; ASM32-NEXT:          .byte   0                               # EOM(1)
; ASM32-NEXT:          .byte   0                               # EOM(2)
; ASM32-NEXT:          .byte   2                               # Abbreviation Code
; ASM32-NEXT:          .byte   46                              # DW_TAG_subprogram
; ASM32-NEXT:          .byte   0                               # DW_CHILDREN_no
; ASM32-NEXT:          .byte   17                              # DW_AT_low_pc
; ASM32-NEXT:          .byte   1                               # DW_FORM_addr
; ASM32-NEXT:          .byte   18                              # DW_AT_high_pc
; ASM32-NEXT:          .byte   6                               # DW_FORM_data4
; ASM32-NEXT:          .byte   64                              # DW_AT_frame_base
; ASM32-NEXT:          .byte   24                              # DW_FORM_exprloc
; ASM32-NEXT:          .byte   3                               # DW_AT_name
; ASM32-NEXT:          .byte   14                              # DW_FORM_strp
; ASM32-NEXT:          .byte   58                              # DW_AT_decl_file
; ASM32-NEXT:          .byte   11                              # DW_FORM_data1
; ASM32-NEXT:          .byte   59                              # DW_AT_decl_line
; ASM32-NEXT:          .byte   11                              # DW_FORM_data1
; ASM32-NEXT:          .byte   39                              # DW_AT_prototyped
; ASM32-NEXT:          .byte   25                              # DW_FORM_flag_present
; ASM32-NEXT:          .byte   73                              # DW_AT_type
; ASM32-NEXT:          .byte   19                              # DW_FORM_ref4
; ASM32-NEXT:          .byte   63                              # DW_AT_external
; ASM32-NEXT:          .byte   25                              # DW_FORM_flag_present
; ASM32-NEXT:          .byte   0                               # EOM(1)
; ASM32-NEXT:          .byte   0                               # EOM(2)
; ASM32-NEXT:          .byte   3                               # Abbreviation Code
; ASM32-NEXT:          .byte   36                              # DW_TAG_base_type
; ASM32-NEXT:          .byte   0                               # DW_CHILDREN_no
; ASM32-NEXT:          .byte   3                               # DW_AT_name
; ASM32-NEXT:          .byte   14                              # DW_FORM_strp
; ASM32-NEXT:          .byte   62                              # DW_AT_encoding
; ASM32-NEXT:          .byte   11                              # DW_FORM_data1
; ASM32-NEXT:          .byte   11                              # DW_AT_byte_size
; ASM32-NEXT:          .byte   11                              # DW_FORM_data1
; ASM32-NEXT:          .byte   0                               # EOM(1)
; ASM32-NEXT:          .byte   0                               # EOM(2)
; ASM32-NEXT:          .byte   0                               # EOM(3)
; ASM32:               .dwsect 0x10000
; ASM32-NEXT:  L...dwinfo:
; ASM32-NEXT:  L..cu_begin0:
; ASM32-NEXT:          .vbyte  2, 4                            # DWARF version number
; ASM32-NEXT:          .vbyte  4, L...dwabrev                  # Offset Into Abbrev. Section
; ASM32-NEXT:          .byte   4                               # Address Size (in bytes)
; ASM32-NEXT:          .byte   1                               # Abbrev [1] 0xb:0x38 DW_TAG_compile_unit
; ASM32-NEXT:          .vbyte  4, L..info_string0              # DW_AT_producer
; ASM32-NEXT:          .vbyte  2, 12                           # DW_AT_language
; ASM32-NEXT:          .vbyte  4, L..info_string1              # DW_AT_name
; ASM32-NEXT:          .vbyte  4, L..line_table_start0         # DW_AT_stmt_list
; ASM32-NEXT:          .vbyte  4, L..info_string2              # DW_AT_comp_dir
; ASM32-NEXT:          .vbyte  4, L..func_begin0               # DW_AT_low_pc
; ASM32-NEXT:          .vbyte  4, L..func_end0-L..func_begin0  # DW_AT_high_pc
; ASM32-NEXT:          .byte   2                               # Abbrev [2] 0x26:0x15 DW_TAG_subprogram
; ASM32-NEXT:          .vbyte  4, L..func_begin0               # DW_AT_low_pc
; ASM32-NEXT:          .vbyte  4, L..func_end0-L..func_begin0  # DW_AT_high_pc
; ASM32-NEXT:          .byte   1                               # DW_AT_frame_base
; ASM32-NEXT:          .byte   81
; ASM32-NEXT:          .vbyte  4, L..info_string3              # DW_AT_name
; ASM32-NEXT:          .byte   1                               # DW_AT_decl_file
; ASM32-NEXT:          .byte   1                               # DW_AT_decl_line
; ASM32-NEXT:                                          # DW_AT_prototyped
; ASM32-NEXT:          .vbyte  4, 59                           # DW_AT_type
; ASM32-NEXT:                                          # DW_AT_external
; ASM32-NEXT:          .byte   3                               # Abbrev [3] 0x3b:0x7 DW_TAG_base_type
; ASM32-NEXT:          .vbyte  4, L..info_string4              # DW_AT_name
; ASM32-NEXT:          .byte   5                               # DW_AT_encoding
; ASM32-NEXT:          .byte   4                               # DW_AT_byte_size
; ASM32-NEXT:          .byte   0                               # End Of Children Mark
; ASM32-NEXT:  L..debug_info_end0:
; ASM32:               .dwsect 0x70000
; ASM32-NEXT:  L...dwstr:
; ASM32-NEXT:  L..info_string0:
; ASM32-NEXT:          .string "clang version 12.0.0"          # string offset=0
; ASM32-NEXT:  L..info_string1:
; ASM32-NEXT:          .string "1.c"                           # string offset=21
; ASM32-NEXT:  L..info_string2:
; ASM32-NEXT:          .string "debug"                         # string offset=25
; ASM32-NEXT:  L..info_string3:
; ASM32-NEXT:          .string "main"                          # string offset=31
; ASM32-NEXT:  L..info_string4:
; ASM32-NEXT:          .string "int"                           # string offset=36
; ASM32-NEXT:          .toc
; ASM32:               .dwsect 0x20000
; ASM32-NEXT:  L...dwline:
; ASM32-NEXT:  L..debug_line_0:
; ASM32-NEXT:  .set L..line_table_start0, L..debug_line_0-4
; ASM32-NEXT:          .vbyte  2, 4
; ASM32-NEXT:          .vbyte	4, L..prologue_end0-L..prologue_start0
; ASM32-NEXT:  L..prologue_start0:
; ASM32-NEXT:          .byte   4
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   -5
; ASM32-NEXT:          .byte   14
; ASM32-NEXT:          .byte   13
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   "debug"
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   "1.c"
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:          .byte   0
; ASM32-NEXT:  L..prologue_end0:
; ASM32-NEXT:          .byte   0                               # Set address to L..tmp0
; ASM32-NEXT:          .byte   5
; ASM32-NEXT:          .byte   2
; ASM32-NEXT:          .vbyte  4, L..tmp0
; ASM32-NEXT:          .byte   19                              # Start sequence
; ASM32-NEXT:          .byte   5
; ASM32-NEXT:          .byte   3
; ASM32-NEXT:          .byte   10
; ASM32-NEXT:          .byte   0                               # Set address to L..tmp2
; ASM32-NEXT:          .byte   5
; ASM32-NEXT:          .byte   2
; ASM32-NEXT:          .vbyte  4, L..tmp2
; ASM32-NEXT:          .byte   3                               # Advance line 1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   0                               # Set address to L..sec_end0
; ASM32-NEXT:          .byte   5
; ASM32-NEXT:          .byte   2
; ASM32-NEXT:          .vbyte  4, L..sec_end0
; ASM32-NEXT:          .byte   0                               # End sequence
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:          .byte   1
; ASM32-NEXT:  L..debug_line_end0:

; ASM64:               .csect .text[PR],2
; ASM64-NEXT:          .file   "1.c"
; ASM64-NEXT:          .globl  main[DS]                        # -- Begin function main
; ASM64-NEXT:          .globl  .main
; ASM64-NEXT:          .align  2
; ASM64-NEXT:          .csect main[DS],3
; ASM64-NEXT:          .vbyte  8, .main                        # @main
; ASM64-NEXT:          .vbyte  8, TOC[TC0]
; ASM64-NEXT:          .vbyte  8, 0
; ASM64-NEXT:          .csect .text[PR],2
; ASM64-NEXT:  .main:
; ASM64-NEXT:  L..func_begin0:
; ASM64-NEXT:  # %bb.0:                                # %entry
; ASM64-NEXT:  L..tmp0:
; ASM64-NEXT:          li 4, 0
; ASM64-NEXT:  L..tmp1:
; ASM64-NEXT:  L..tmp2:
; ASM64-NEXT:          li 3, 0
; ASM64-NEXT:          stw 4, -4(1)
; ASM64-NEXT:          blr
; ASM64-NEXT:  L..tmp3:
; ASM64-NEXT:  L..main0:
; ASM64-NEXT:          .vbyte  4, 0x00000000                   # Traceback table begin
; ASM64-NEXT:          .byte   0x00                            # Version = 0
; ASM64-NEXT:          .byte   0x09                            # Language = CPlusPlus
; ASM64-NEXT:          .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; ASM64-NEXT:                                          # +HasTraceBackTableOffset, -IsInternalProcedure
; ASM64-NEXT:                                          # -HasControlledStorage, -IsTOCless
; ASM64-NEXT:                                          # -IsFloatingPointPresent
; ASM64-NEXT:                                          # -IsFloatingPointOperationLogOrAbortEnabled
; ASM64-NEXT:          .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; ASM64-NEXT:                                          # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; ASM64-NEXT:          .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; ASM64-NEXT:          .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; ASM64-NEXT:          .byte   0x00                            # NumberOfFixedParms = 0
; ASM64-NEXT:          .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; ASM64-NEXT:          .vbyte  4, L..main0-.main               # Function size
; ASM64-NEXT:          .vbyte  2, 0x0004                       # Function name len = 4
; ASM64-NEXT:          .byte   "main"                          # Function Name
; ASM64-NEXT:  L..func_end0:
; ASM64-NEXT:                                          # -- End function
; ASM64-NEXT:  L..sec_end0:
; ASM64:               .dwsect 0x60000
; ASM64-NEXT:  L...dwabrev:
; ASM64-NEXT:          .byte   1                               # Abbreviation Code
; ASM64-NEXT:          .byte   17                              # DW_TAG_compile_unit
; ASM64-NEXT:          .byte   1                               # DW_CHILDREN_yes
; ASM64-NEXT:          .byte   37                              # DW_AT_producer
; ASM64-NEXT:          .byte   14                              # DW_FORM_strp
; ASM64-NEXT:          .byte   19                              # DW_AT_language
; ASM64-NEXT:          .byte   5                               # DW_FORM_data2
; ASM64-NEXT:          .byte   3                               # DW_AT_name
; ASM64-NEXT:          .byte   14                              # DW_FORM_strp
; ASM64-NEXT:          .byte   16                              # DW_AT_stmt_list
; ASM64-NEXT:          .byte   23                              # DW_FORM_sec_offset
; ASM64-NEXT:          .byte   27                              # DW_AT_comp_dir
; ASM64-NEXT:          .byte   14                              # DW_FORM_strp
; ASM64-NEXT:          .byte   17                              # DW_AT_low_pc
; ASM64-NEXT:          .byte   1                               # DW_FORM_addr
; ASM64-NEXT:          .byte   18                              # DW_AT_high_pc
; ASM64-NEXT:          .byte   6                               # DW_FORM_data4
; ASM64-NEXT:          .byte   0                               # EOM(1)
; ASM64-NEXT:          .byte   0                               # EOM(2)
; ASM64-NEXT:          .byte   2                               # Abbreviation Code
; ASM64-NEXT:          .byte   46                              # DW_TAG_subprogram
; ASM64-NEXT:          .byte   0                               # DW_CHILDREN_no
; ASM64-NEXT:          .byte   17                              # DW_AT_low_pc
; ASM64-NEXT:          .byte   1                               # DW_FORM_addr
; ASM64-NEXT:          .byte   18                              # DW_AT_high_pc
; ASM64-NEXT:          .byte   6                               # DW_FORM_data4
; ASM64-NEXT:          .byte   64                              # DW_AT_frame_base
; ASM64-NEXT:          .byte   24                              # DW_FORM_exprloc
; ASM64-NEXT:          .byte   3                               # DW_AT_name
; ASM64-NEXT:          .byte   14                              # DW_FORM_strp
; ASM64-NEXT:          .byte   58                              # DW_AT_decl_file
; ASM64-NEXT:          .byte   11                              # DW_FORM_data1
; ASM64-NEXT:          .byte   59                              # DW_AT_decl_line
; ASM64-NEXT:          .byte   11                              # DW_FORM_data1
; ASM64-NEXT:          .byte   39                              # DW_AT_prototyped
; ASM64-NEXT:          .byte   25                              # DW_FORM_flag_present
; ASM64-NEXT:          .byte   73                              # DW_AT_type
; ASM64-NEXT:          .byte   19                              # DW_FORM_ref4
; ASM64-NEXT:          .byte   63                              # DW_AT_external
; ASM64-NEXT:          .byte   25                              # DW_FORM_flag_present
; ASM64-NEXT:          .byte   0                               # EOM(1)
; ASM64-NEXT:          .byte   0                               # EOM(2)
; ASM64-NEXT:          .byte   3                               # Abbreviation Code
; ASM64-NEXT:          .byte   36                              # DW_TAG_base_type
; ASM64-NEXT:          .byte   0                               # DW_CHILDREN_no
; ASM64-NEXT:          .byte   3                               # DW_AT_name
; ASM64-NEXT:          .byte   14                              # DW_FORM_strp
; ASM64-NEXT:          .byte   62                              # DW_AT_encoding
; ASM64-NEXT:          .byte   11                              # DW_FORM_data1
; ASM64-NEXT:          .byte   11                              # DW_AT_byte_size
; ASM64-NEXT:          .byte   11                              # DW_FORM_data1
; ASM64-NEXT:          .byte   0                               # EOM(1)
; ASM64-NEXT:          .byte   0                               # EOM(2)
; ASM64-NEXT:          .byte   0                               # EOM(3)
; ASM64:               .dwsect 0x10000
; ASM64-NEXT:  L...dwinfo:
; ASM64-NEXT:  L..cu_begin0:
; ASM64-NEXT:          .vbyte  2, 4                            # DWARF version number
; ASM64-NEXT:          .vbyte  8, L...dwabrev                  # Offset Into Abbrev. Section
; ASM64-NEXT:          .byte   8                               # Address Size (in bytes)
; ASM64-NEXT:          .byte   1                               # Abbrev [1] 0x17:0x58 DW_TAG_compile_unit
; ASM64-NEXT:          .vbyte  8, L..info_string0              # DW_AT_producer
; ASM64-NEXT:          .vbyte  2, 12                           # DW_AT_language
; ASM64-NEXT:          .vbyte  8, L..info_string1              # DW_AT_name
; ASM64-NEXT:          .vbyte  8, L..line_table_start0         # DW_AT_stmt_list
; ASM64-NEXT:          .vbyte  8, L..info_string2              # DW_AT_comp_dir
; ASM64-NEXT:          .vbyte  8, L..func_begin0               # DW_AT_low_pc
; ASM64-NEXT:          .vbyte  4, L..func_end0-L..func_begin0  # DW_AT_high_pc
; ASM64-NEXT:          .byte   2                               # Abbrev [2] 0x46:0x1d DW_TAG_subprogram
; ASM64-NEXT:          .vbyte  8, L..func_begin0               # DW_AT_low_pc
; ASM64-NEXT:          .vbyte  4, L..func_end0-L..func_begin0  # DW_AT_high_pc
; ASM64-NEXT:          .byte   1                               # DW_AT_frame_base
; ASM64-NEXT:          .byte   81
; ASM64-NEXT:          .vbyte  8, L..info_string3              # DW_AT_name
; ASM64-NEXT:          .byte   1                               # DW_AT_decl_file
; ASM64-NEXT:          .byte   1                               # DW_AT_decl_line
; ASM64-NEXT:                                          # DW_AT_prototyped
; ASM64-NEXT:          .vbyte  4, 99                           # DW_AT_type
; ASM64-NEXT:                                          # DW_AT_external
; ASM64-NEXT:          .byte   3                               # Abbrev [3] 0x63:0xb DW_TAG_base_type
; ASM64-NEXT:          .vbyte  8, L..info_string4              # DW_AT_name
; ASM64-NEXT:          .byte   5                               # DW_AT_encoding
; ASM64-NEXT:          .byte   4                               # DW_AT_byte_size
; ASM64-NEXT:          .byte   0                               # End Of Children Mark
; ASM64-NEXT:  L..debug_info_end0:
; ASM64:               .dwsect 0x70000
; ASM64-NEXT:  L...dwstr:
; ASM64-NEXT:  L..info_string0:
; ASM64-NEXT:          .string "clang version 12.0.0"          # string offset=0
; ASM64-NEXT:  L..info_string1:
; ASM64-NEXT:          .string "1.c"                           # string offset=21
; ASM64-NEXT:  L..info_string2:
; ASM64-NEXT:          .string "debug"                         # string offset=25
; ASM64-NEXT:  L..info_string3:
; ASM64-NEXT:          .string "main"                          # string offset=31
; ASM64-NEXT:  L..info_string4:
; ASM64-NEXT:          .string "int"                           # string offset=36
; ASM64-NEXT:         .toc
; ASM64:               .dwsect 0x20000
; ASM64-NEXT:  L...dwline:
; ASM64-NEXT:  L..debug_line_0:
; ASM64-NEXT:  .set L..line_table_start0, L..debug_line_0-12
; ASM64-NEXT:          .vbyte  2, 4
; ASM64-NEXT:          .vbyte  8, L..prologue_end0-L..prologue_start0
; ASM64-NEXT:  L..prologue_start0:
; ASM64-NEXT:          .byte   4
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   -5
; ASM64-NEXT:          .byte   14
; ASM64-NEXT:          .byte   13
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   "debug"
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   "1.c"
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:          .byte   0
; ASM64-NEXT:  L..prologue_end0:
; ASM64-NEXT:          .byte   0                               # Set address to L..tmp0
; ASM64-NEXT:          .byte   9
; ASM64-NEXT:          .byte   2
; ASM64-NEXT:          .vbyte  8, L..tmp0
; ASM64-NEXT:          .byte   19                              # Start sequence
; ASM64-NEXT:          .byte   5
; ASM64-NEXT:          .byte   3
; ASM64-NEXT:          .byte   10
; ASM64-NEXT:          .byte   0                               # Set address to L..tmp2
; ASM64-NEXT:          .byte   9
; ASM64-NEXT:          .byte   2
; ASM64-NEXT:          .vbyte  8, L..tmp2
; ASM64-NEXT:          .byte   3                               # Advance line 1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   0                               # Set address to L..sec_end0
; ASM64-NEXT:          .byte   9
; ASM64-NEXT:          .byte   2
; ASM64-NEXT:          .vbyte  8, L..sec_end0
; ASM64-NEXT:          .byte   0                               # End sequence
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:          .byte   1
; ASM64-NEXT:  L..debug_line_end0:
