; RUN: opt < %s -passes=reassociate -S | FileCheck %s

; Reduced from issue #60532. The Reassociate pass rearranges the decrement
; of 'day', verify that the debug value for 'day' is salvaged using a
; DIArgList expression rather than dropped to poison.

; CHECK-LABEL: @example(
; CHECK:         #dbg_value(i32 %mday, ![[DAY:[0-9]+]], !DIExpression(),
; CHECK:         #dbg_value(!DIArgList(i32 -1, i32 %mday), ![[DAY]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value),
; CHECK-NOT:     #dbg_value(i32 poison,
; CHECK:       ![[DAY]] = !DILocalVariable(name: "day"

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = internal constant [2 x i32] [i32 9, i32 4], align 4

define dso_local i64 @example(i32 noundef %year, i32 noundef %mon, i32 noundef %mday) !dbg !7 {
entry:
  %idxprom = sext i32 %mon to i64, !dbg !22
  %arrayidx = getelementptr inbounds i32, ptr @b, i64 %idxprom, !dbg !22
  %0 = load i32, ptr %arrayidx, align 4, !dbg !22
    #dbg_value(i32 %mday, !16, !DIExpression(), !23)
  %dec = add nsw i32 %mday, -1, !dbg !24
    #dbg_value(i32 %dec, !16, !DIExpression(), !23)
  %add = add nsw i32 %year, %0, !dbg !25
  %add1 = add nsw i32 %add, %dec, !dbg !26
  %conv = sext i32 %add1 to i64, !dbg !27
  ret i64 %conv, !dbg !28
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "example.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DISubprogram(name: "example", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !6, !6, !6}
!10 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !16}
!12 = !DILocalVariable(name: "year", arg: 1, scope: !7, file: !1, line: 3, type: !6)
!13 = !DILocalVariable(name: "mon", arg: 2, scope: !7, file: !1, line: 3, type: !6)
!14 = !DILocalVariable(name: "mday", arg: 3, scope: !7, file: !1, line: 3, type: !6)
!16 = !DILocalVariable(name: "day", scope: !7, file: !1, line: 6, type: !6)
!22 = !DILocation(line: 8, column: 14, scope: !7)
!23 = !DILocation(line: 6, column: 7, scope: !7)
!24 = !DILocation(line: 7, column: 6, scope: !7)
!25 = !DILocation(line: 8, column: 12, scope: !7)
!26 = !DILocation(line: 8, column: 20, scope: !7)
!27 = !DILocation(line: 8, column: 10, scope: !7)
!28 = !DILocation(line: 8, column: 3, scope: !7)
