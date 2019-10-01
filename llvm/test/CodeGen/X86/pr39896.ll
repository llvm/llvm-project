; RUN: llc %s -start-after=codegenprepare -stop-after=finalize-isel -o - | FileCheck %s

; PR39896: When code such as %conv below is dropped by SelectionDAG for having
; no users, don't just drop the dbg.value record associated with it. Instead,
; the corresponding variable should receive a "DBG_VALUE" undef to terminate
; earlier variable live ranges.

; ModuleID = 'run.c'
source_filename = "run.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@a = global i8 25, align 1, !dbg !0

define signext i16 @b() !dbg !12 {
entry:
; CHECK: DBG_VALUE 23680, $noreg, ![[VARNUM:[0-9]+]],
  call void @llvm.dbg.value(metadata i16 23680, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = load i8, i8* @a, align 1, !dbg !18
  %conv = sext i8 %0 to i16, !dbg !18
; CHECK: DBG_VALUE $noreg, $noreg, ![[VARNUM]],
  call void @llvm.dbg.value(metadata i16 %conv, metadata !17, metadata !DIExpression()), !dbg !18
  %call = call i32 (...) @optimize_me_not(), !dbg !18
  %1 = load i8, i8* @a, align 1, !dbg !18
  %conv1 = sext i8 %1 to i16, !dbg !18
  ret i16 %conv1, !dbg !18
}
declare i32 @optimize_me_not(...)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 348101) (llvm/trunk 348109)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "run.c", directory: "/Users/dcci/work/llvm/build-debug/bin")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version something"}
!12 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !13, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DILocalVariable(name: "i", scope: !12, file: !3, line: 3, type: !15)
!18 = !DILocation(line: 3, scope: !12)
