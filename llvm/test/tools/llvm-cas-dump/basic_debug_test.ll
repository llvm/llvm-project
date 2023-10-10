; RUN: rm -rf %t.casdb
; RUN: llc -cas-friendly-debug-info -O0 --filetype=obj --cas-backend --cas=%t.casdb --mccas-casid -o %t.casid %s
; RUN: llvm-cas-dump --cas=%t.casdb --dwarf-sections-only --dwarf-dump --casid-file %t.casid | FileCheck %s --check-prefix=DWARF
; RUN: llvm-cas-dump --cas=%t.casdb --dwarf-sections-only --casid-file %t.casid | FileCheck %s
; RUN: llc -cas-friendly-debug-info -O0 --filetype=obj --cas-backend --cas=%t.casdb --mccas-casid -o %t_DIE.casid %s
; RUN: llvm-cas-dump --cas=%t.casdb --casid-file %t_DIE.casid  --die-refs | FileCheck %s --check-prefix=DWARF-DIE

; REQUIRES: aarch64-registered-target

; This test is created from a C program like:
; int foo() { return 10; }

; DWARF: mc:debug_line_section
; DWARF:   debug_line[0x00000000]
; DWARF:   Line table prologue:
; DWARF:       total_length: 0x000000{{[0-9a-f]+}}
; DWARF:             format: DWARF32
; DWARF:            version: 4
; DWARF:    prologue_length: 0x000000{{[0-9a-f]+}}
; DWARF: 0x0000000000000000      3      3      1   0             0       0  is_stmt prologue_end
; DWARF: 0x0000000000000008      3      3      1   0             0       0  is_stmt end_sequence

; CHECK: CASID File Name: {{.+}}
; CHECK-NEXT:      mc:assembler  llvmcas://{{.*}}
; CHECK-NEXT:   mc:header  llvmcas://{{.*}}
; CHECK-NEXT:   mc:group  llvmcas://{{.*}}
; CHECK-NEXT:     mc:debug_abbrev_section  llvmcas://{{.*}}
; CHECK-NEXT:       mc:padding  llvmcas://{{.*}}
; CHECK-NEXT:     mc:debug_info_section  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_DIE_top_level  llvmcas://{{.*}}
; CHECK-NEXT:         mc:debug_DIE_data       llvmcas://{{.*}}
; CHECK-NEXT:         mc:debug_DIE_data       llvmcas://{{.*}}
; CHECK-NEXT:         mc:debug_DIE_abbrev_set llvmcas://{{.*}}
; CHECK-NEXT:           mc:debug_DIE_abbrev     llvmcas://{{.*}}
; CHECK-NEXT:           mc:debug_DIE_abbrev     llvmcas://{{.*}}
; CHECK-NEXT:           mc:debug_DIE_abbrev     llvmcas://{{.*}}
; CHECK-NEXT:         mc:debug_DIE_distinct_data       llvmcas://{{.*}}
; CHECK-NEXT:       mc:padding  llvmcas://{{.*}}
; CHECK-NEXT:     mc:debug_string_section  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_string  llvmcas://{{.*}}
; CHECK-NEXT:       mc:padding  llvmcas://{{.*}}
; CHECK-NEXT:     mc:debug_line_section  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_line_distinct_data  llvmcas://{{.*}}
; CHECK-NEXT:       mc:debug_line  llvmcas://{{.*}}
; CHECK-NEXT:       mc:padding  llvmcas://{{.*}}
; CHECK-NEXT:   mc:data_in_code  llvmcas://{{.*}}
; CHECK-NEXT:   mc:symbol_table  llvmcas://{{.*}}
; CHECK-NEXT:     mc:cstring  llvmcas://{{.*}}
; CHECK-NEXT:     mc:cstring  llvmcas://{{.*}}
; CHECK-NEXT:     mc:cstring  llvmcas://{{.*}}
; CHECK-NEXT:     mc:cstring  llvmcas://{{.*}}

; DWARF-DIE:        mc:debug_DIE_top_level llvmcas://{{.*}}
; DWARF-DIE-NEXT:   Header = [4B 0 0 0 4 0 0 0 0 0 8]
; DWARF-DIE-NEXT:   CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_compile_unit  AbbrevIdx = 2
; DWARF-DIE-NEXT:     DW_AT_producer                 DW_FORM_strp_cas       [distinct]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_language                 DW_FORM_data2          [dedups]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_name                     DW_FORM_strp_cas       [distinct]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_LLVM_sysroot             DW_FORM_strp_cas       [distinct]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_stmt_list                DW_FORM_sec_offset     [dedups]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_comp_dir                 DW_FORM_strp_cas       [distinct]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_low_pc                   DW_FORM_addr           [distinct]    [{{.*}}]
; DWARF-DIE-NEXT:     DW_AT_high_pc                  DW_FORM_data4          [dedups]    [{{.*}}]
; DWARF-DIE-NEXT:     CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:     DW_TAG_subprogram  AbbrevIdx = 3
; DWARF-DIE-NEXT:       DW_AT_low_pc                   DW_FORM_addr         [distinct]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_high_pc                  DW_FORM_data4        [dedups]     [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_APPLE_omit_frame_ptr     DW_FORM_flag_present [dedups]      []
; DWARF-DIE-NEXT:       DW_AT_frame_base               DW_FORM_exprloc      [dedups]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_name                     DW_FORM_strp_cas     [distinct]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_decl_file                DW_FORM_data1        [distinct]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_decl_line                DW_FORM_data1        [dedups]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_type                     DW_FORM_ref4_cas     [distinct]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_external                 DW_FORM_flag_present [dedups]      []
; DWARF-DIE-NEXT:     DW_TAG_base_type  AbbrevIdx = 4
; DWARF-DIE-NEXT:       DW_AT_name                     DW_FORM_strp_cas     [distinct]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_encoding                 DW_FORM_data1        [dedups]      [{{.*}}]
; DWARF-DIE-NEXT:       DW_AT_byte_size                DW_FORM_data1        [dedups]      [{{.*}}]

target triple = "arm64-apple-macosx12.0.0"

define i32 @foo() !dbg !9 {
entry:
  ret i32 10, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "some_clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "test.c", directory: "some_dir")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"some_clang"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocation(line: 3, column: 3, scope: !9)
