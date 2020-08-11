; This test checks emission of the GNU extension for the .debug_macro section.

; RUN: %llc_dwarf -dwarf-version=4 -O0 -use-gnu-debug-macro -filetype=obj < %s | llvm-dwarfdump -v - | FileCheck %s

; CHECK-LABEL:  .debug_info contents:
; CHECK: DW_AT_GNU_macros [DW_FORM_sec_offset] (0x00000000)

; CHECK-LABEL:  .debug_macro contents:
; CHECK-NEXT: 0x00000000:
; CHECK-NEXT: macro header: version = 0x0004, flags = 0x02, format = DWARF32, debug_line_offset = 0x0000
; CHECK-NEXT: DW_MACRO_GNU_start_file - lineno: 0 filenum: 1
; CHECK-NEXT:   DW_MACRO_GNU_start_file - lineno: 1 filenum: 2
; CHECK-NEXT:     DW_MACRO_GNU_define_indirect - lineno: 1 macro: FOO 5
; CHECK-NEXT:   DW_MACRO_GNU_end_file
; CHECK-NEXT:   DW_MACRO_GNU_start_file - lineno: 2 filenum: 3
; CHECK-NEXT:     DW_MACRO_GNU_undef_indirect - lineno: 14 macro: YEA
; CHECK-NEXT:   DW_MACRO_GNU_end_file
; CHECK-NEXT:   DW_MACRO_GNU_undef_indirect - lineno: 14 macro: YEA
; CHECK-NEXT: DW_MACRO_GNU_end_file

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p200:32:32-p201:32:32-p202:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, macros: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/")
!2 = !{}
!3 = !{!4}
!4 = !DIMacroFile(file: !1, nodes: !5)
!5 = !{!6, !10, !13}
!6 = !DIMacroFile(line: 1, file: !7, nodes: !8)
!7 = !DIFile(filename: "./foo.h", directory: "/home/")
!8 = !{!9}
!9 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "FOO", value: "5")
!10 = !DIMacroFile(line: 2, file: !11, nodes: !12)
!11 = !DIFile(filename: "./bar.h", directory: "/home/")
!12 = !{!13}
!13 = !DIMacro(type: DW_MACINFO_undef, line: 14, name: "YEA")
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 10.0.0"}
