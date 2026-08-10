; RUN: llc -O0 -split-dwarf-file=foo.dwo -filetype=obj < %s | llvm-dwarfdump --debug-info --debug-macro - | FileCheck %s

; CHECK:.debug_info.dwo contents:
; CHECK:     DW_TAG_compile_unit
; CHECK-NOT:   DW_TAG
; CHECK:       DW_AT_GNU_dwo_name ("foo.dwo")
; CHECK-NOT:   DW_TAG
; CHECK:       DW_AT_macro_info  (0x00000000)

;CHECK-LABEL:.debug_macinfo.dwo contents:
;CHECK-NEXT: 0x00000000:
;CHECK-NEXT:  DW_MACINFO_start_file - lineno: 0 filenum: 1
;CHECK-NEXT:    DW_MACINFO_start_file - lineno: 1 filenum: 2
;CHECK-NEXT:      DW_MACINFO_define - lineno: 1 macro: define_1 12
;CHECK-NEXT:    DW_MACINFO_end_file
;CHECK-NEXT:    DW_MACINFO_start_file - lineno: 2 filenum: 3
;CHECK-NEXT:      DW_MACINFO_define - lineno: 1 macro: define_2 14
;CHECK-NEXT:    DW_MACINFO_end_file
;CHECK-NEXT:  DW_MACINFO_end_file
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __llvm__ 1
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __clang__ 1
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __clang_major__ 10
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __clang_minor__ 0
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __clang_patchlevel__ 0
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __clang_version__ 10.0.0
;CHECK-NEXT:  DW_MACINFO_define - lineno: 0 macro: __GNUC__ 4

; ModuleID = 'debug-macro-split-dwarf.c'
source_filename = "debug-macro-split-dwarf.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !25 {
entry:
  ret void, !dbg !34
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "debug-macro-split-dwarf.dwo", emissionKind: FullDebug, enums: !2, macros: !3, nameTableKind: GNU)
!1 = !DIFile(filename: "debug-macro-split-dwarf.c", directory: "/", checksumkind: CSK_MD5, checksum: "e74d0fa8f714535c1bac6da2ffbbd898")
!2 = !{}
!3 = !{!4, !14, !15, !16, !17, !18, !19, !20}
!4 = !DIMacroFile(file: !1, nodes: !5)
!5 = !{!6, !10}
!6 = !DIMacroFile(line: 1, file: !7, nodes: !8)
!7 = !DIFile(filename: "./1.h", directory: "/", checksumkind: CSK_MD5, checksum: "6185a3a5ae6eb7d1fd2692718f9d95e5")
!8 = !{!9}
!9 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "define_1", value: "12")
!10 = !DIMacroFile(line: 2, file: !11, nodes: !12)
!11 = !DIFile(filename: "./2.h", directory: "/", checksumkind: CSK_MD5, checksum: "d48d124c86c1b50a32517884ff962f83")
!12 = !{!13}
!13 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "define_2", value: "14")
!14 = !DIMacro(type: DW_MACINFO_define, name: "__llvm__", value: "1")
!15 = !DIMacro(type: DW_MACINFO_define, name: "__clang__", value: "1")
!16 = !DIMacro(type: DW_MACINFO_define, name: "__clang_major__", value: "10")
!17 = !DIMacro(type: DW_MACINFO_define, name: "__clang_minor__", value: "0")
!18 = !DIMacro(type: DW_MACINFO_define, name: "__clang_patchlevel__", value: "0")
!19 = !DIMacro(type: DW_MACINFO_define, name: "__clang_version__", value: "10.0.0")
!20 = !DIMacro(type: DW_MACINFO_define, name: "__GNUC__", value: "4")
!21 = !{i32 7, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 4}
!24 = !{!"clang version 10.0.0 "}
!25 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !26, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null}
!30 = !DILocation(line: 4, column: 14, scope: !25)
!32 = !DILocation(line: 4, column: 21, scope: !25)
!33 = !DILocation(line: 5, column: 4, scope: !25)
!34 = !DILocation(line: 6, column: 1, scope: !25)
