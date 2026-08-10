; This checks that .debug_macro[.dwo] can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=4 -dwarf64 -use-gnu-debug-macro -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-macro %t | FileCheck %s --check-prefix=DWARF4

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -dwarf64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-macro %t | FileCheck %s --check-prefix=DWARF5

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -dwarf64 -split-dwarf-file=foo.dwo -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-macro %t | FileCheck %s --check-prefixes=DWARF5,DWO

; DWARF4:      .debug_macro contents:
; DWARF4-NEXT: 0x00000000:
; DWARF4-NEXT: macro header: version = 0x0004, flags = 0x03, format = DWARF64, debug_line_offset = 0x0000000000000000
; DWARF4-NEXT: DW_MACRO_GNU_start_file - lineno: 0 filenum: 1
; DWARF4-NEXT:  DW_MACRO_GNU_define_indirect - lineno: 1 macro: FOO 1
; DWARF4-NEXT:  DW_MACRO_GNU_undef_indirect - lineno: 2 macro: BAR
; DWARF4-NEXT: DW_MACRO_GNU_end_file

; DWARF5:      .debug_macro contents:
; DWO:         .debug_macro.dwo contents:
; DWARF5-NEXT: 0x00000000:
; DWARF5-NEXT: macro header: version = 0x0005, flags = 0x03, format = DWARF64, debug_line_offset = 0x0000000000000000
; DWARF5-NEXT: DW_MACRO_start_file - lineno: 0 filenum: 0
; DWARF5-NEXT:  DW_MACRO_define_strx - lineno: 1 macro: FOO 1
; DWARF5-NEXT:  DW_MACRO_undef_strx - lineno: 2 macro: BAR
; DWARF5-NEXT: DW_MACRO_end_file

; IR generated and reduced from:
; $ cat foo.c
; #define FOO 1
; #undef BAR
; $ clang -g -S -emit-llvm -fdebug-macro foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!348, !349, !350}
!llvm.ident = !{!351}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, macros: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIMacroFile(file: !1, nodes: !5)
!5 = !{!6, !7}
!6 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "FOO", value: "1")
!7 = !DIMacro(type: DW_MACINFO_undef, line: 2, name: "BAR")
!348 = !{i32 7, !"Dwarf Version", i32 4}
!349 = !{i32 2, !"Debug Info Version", i32 3}
!350 = !{i32 1, !"wchar_size", i32 4}
!351 = !{!"clang version 12.0.0"}
