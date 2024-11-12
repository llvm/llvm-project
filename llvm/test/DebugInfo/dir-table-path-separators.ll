; RUN: llc %s -o - -filetype=obj -mtriple x86_64-pc-linux-gnu | llvm-dwarfdump - --debug-line | FileCheck %s --check-prefix=LINUX
; RUN: llc %s -o - -filetype=obj -mtriple x86_64-pc-linux-gnu -force-dwarf-windows-path-seps=true | llvm-dwarfdump - --debug-line | FileCheck %s --check-prefix=PS5
; RUN: llc %s -o - -filetype=obj -mtriple x86_64-sie-ps5 | llvm-dwarfdump - --debug-line | FileCheck %s --check-prefix=PS5
;
; UNSUPPORTED: system-windows
;
; Check that the DWARF-printing MC backend is willing to consider Windows '\'
; characters as path separators so that it can build the directory index table.
; On Linux, the Windows path separators below would been seen as part of the
; filename, and so wouldn't be combined into a directory entry. Wheras on
; Windows (or a target masquerading as Windows) they should be combined into a
; "foo\bar" directory.
;
; LINUX: include_directories[  0] = "C:\\foobar"
; LINUX: file_names[  0]:
; LINUX:           name: "foo\\bar\\test.cpp"
; LINUX:      dir_index: 0
; LINUX: file_names[  1]:
; LINUX:           name: "foo\\bar\\bar.cpp"
; LINUX:      dir_index: 0
; LINUX: file_names[  2]:
; LINUX:           name: "foo\\bar\\baz.cpp"
; LINUX:      dir_index: 0
;
; PS5:      include_directories[  0] = "C:\\foobar"
; PS5-NEXT: include_directories[  1] = "foo\\bar"
; PS5-NEXT: file_names[  0]:
; PS5-NEXT:           name: "foo\\bar\\test.cpp"
; PS5-NEXT:      dir_index: 0
; PS5-NEXT: file_names[  1]:
; PS5-NEXT:           name: "bar.cpp"
; PS5-NEXT:      dir_index: 1
; PS5-NEXT: file_names[  2]:
; PS5-NEXT:           name: "baz.cpp"
; PS5-NEXT:      dir_index: 1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define dso_local noundef i32 @_Z3foov() local_unnamed_addr !dbg !9 {
  ret i32 0, !dbg !14
}

define dso_local noundef i32 @_Z3barv() local_unnamed_addr !dbg !15 {
  ret i32 0, !dbg !17
}

define dso_local noundef i32 @main() local_unnamed_addr !dbg !18 {
  ret i32 0, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo\\bar\\test.cpp", directory: "C:\\foobar")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DIFile(filename: "foo\\bar\\bar.cpp", directory: "C:\\foobar")
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, column: 3, scope: !9)
!15 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !16, file: !16, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DIFile(filename: "foo\\bar\\baz.cpp", directory: "C:\\foobar")
!17 = !DILocation(line: 2, column: 3, scope: !15)
!18 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = !DILocation(line: 4, column: 3, scope: !18)
