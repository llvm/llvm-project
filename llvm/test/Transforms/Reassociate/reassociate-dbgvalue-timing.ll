; RUN: opt < %s -passes=reassociate -S | FileCheck %s

; Reduced from issue #212714. Reassociation rewrites tmp + 10 + 20 and moves
; the reused instructions away from their original source operations. Ensure
; that their now-stale debug locations are dropped.

; CHECK-LABEL: @example(
; CHECK-NEXT:  entry:
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP:[0-9]+]], !DIExpression(),
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP]], !DIExpression(DW_OP_plus_uconst, 30, DW_OP_stack_value),
; CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 %b to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 [[IDXPROM]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[ARRAYIDX]]
; CHECK-NEXT:    [[ADD:%.*]] = add i32 %a, 30{{$}}
; CHECK-NEXT:    [[ADD1:%.*]] = add i32 [[ADD]], %c{{$}}
; CHECK-NEXT:    [[RESULT:%.*]] = add i32 [[ADD1]], [[LOAD]]
; CHECK-NEXT:    ret i32 [[RESULT]]
; CHECK:       ![[TMP]] = !DILocalVariable(name: "tmp"

@g = internal constant [2 x i32] [i32 9, i32 4], align 4

define i32 @example(i32 %a, i32 %b, i32 %c) !dbg !5 {
entry:
    #dbg_value(i32 %c, !10, !DIExpression(), !11)
  %add = add nsw i32 %c, 10, !dbg !12
  %add1 = add nsw i32 %add, 20, !dbg !13
    #dbg_value(i32 %add1, !10, !DIExpression(), !11)
  %idxprom = sext i32 %b to i64, !dbg !14
  %arrayidx = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 %idxprom, !dbg !14
  %load = load i32, ptr %arrayidx, align 4, !dbg !14
  %add2 = add nsw i32 %a, %load, !dbg !15
  %result = add nsw i32 %add2, %add1, !dbg !16
  ret i32 %result, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "case.c", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "example", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!2, !2, !2, !2}
!10 = !DILocalVariable(name: "tmp", scope: !5, file: !1, line: 5, type: !2)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocation(line: 6, column: 13, scope: !5)
!13 = !DILocation(line: 6, column: 18, scope: !5)
!14 = !DILocation(line: 7, column: 14, scope: !5)
!15 = !DILocation(line: 7, column: 12, scope: !5)
!16 = !DILocation(line: 7, column: 19, scope: !5)
!17 = !DILocation(line: 7, column: 3, scope: !5)
