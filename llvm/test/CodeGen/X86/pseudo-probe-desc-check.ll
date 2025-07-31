; REQUIRES: asserts
; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-windows-msvc < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: warning: Guid:8314849053352128226 Name:inlinee does not exist in pseudo probe desc
; CHECK: warning: Guid:6492337042787843907 Name:extract2 does not exist in pseudo probe desc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @extract1() !dbg !8 {
entry:
  call void @llvm.pseudoprobe(i64 6028998432455395745, i64 1, i32 0, i64 -1), !dbg !11
  call void @llvm.pseudoprobe(i64 8314849053352128226, i64 1, i32 0, i64 -1), !dbg !12
  ret void, !dbg !16
}

define void @extract2() !dbg !17 {
entry:
  call void @llvm.pseudoprobe(i64 6492337042787843907, i64 1, i32 0, i64 -1), !dbg !18
  ret void, !dbg !18
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.pseudo_probe_desc = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/foo")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{i64 6028998432455395745, i64 281479271677951, !"extract1"}
!8 = distinct !DISubprogram(name: "extract1", scope: !1, file: !1, line: 4, type: !9, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{}
!11 = !DILocation(line: 5, column: 3, scope: !8)
!12 = !DILocation(line: 2, column: 1, scope: !13, inlinedAt: !14)
!13 = distinct !DISubprogram(name: "inlinee", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!14 = distinct !DILocation(line: 5, column: 3, scope: !15)
!15 = !DILexicalBlockFile(scope: !8, file: !1, discriminator: 455082007)
!16 = !DILocation(line: 6, column: 1, scope: !8)
!17 = distinct !DISubprogram(name: "extract2", scope: !1, file: !1, line: 8, type: !9, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DILocation(line: 9, column: 1, scope: !17)
