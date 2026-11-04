; RUN: opt < %s -passes=reassociate -S | FileCheck %s

; Reduced from issue #212714. Reassociation moves the line 6 expression after
; part of line 7. Preserve the salvaged tmp update at its original program point,
; restore tmp's old value for the moved expression, then reapply the update.

; CHECK-LABEL: @example(
; CHECK-NEXT:  entry:
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP:[0-9]+]], !DIExpression(),
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP]], !DIExpression(DW_OP_plus_uconst, 30, DW_OP_stack_value),
; CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 %b to i64, !dbg [[LINE7A:![0-9]+]]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 [[IDXPROM]], !dbg [[LINE7A]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[ARRAYIDX]], align 4, !dbg [[LINE7A]]
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP]], !DIExpression(),
; CHECK-NEXT:    [[ADD:%.*]] = add i32 %a, 30, !dbg [[LINE6A:![0-9]+]]
; CHECK-NEXT:    [[ADD1:%.*]] = add i32 [[ADD]], %c, !dbg [[LINE6B:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i32 %c, ![[TMP]], !DIExpression(DW_OP_plus_uconst, 30, DW_OP_stack_value),
; CHECK-NEXT:    [[RESULT:%.*]] = add i32 [[ADD1]], [[LOAD]], !dbg [[LINE7B:![0-9]+]]
; CHECK-NEXT:    ret i32 [[RESULT]], !dbg [[LINE7C:![0-9]+]]

; Reassociation can also move the expression from a predecessor into the root's
; block. Do not add block-local variable view transitions in that case.

; CHECK-LABEL: @across_basic_blocks(
; CHECK-NEXT:  entry:
; CHECK-NEXT:      #dbg_value(i32 %c, ![[CROSS_TMP:[0-9]+]], !DIExpression(),
; CHECK-NEXT:      #dbg_value(i32 %c, ![[CROSS_TMP]], !DIExpression(DW_OP_plus_uconst, 30, DW_OP_stack_value),
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    [[CROSS_IDXPROM:%.*]] = sext i32 %b to i64, !dbg [[CROSS_LINE7A:![0-9]+]]
; CHECK-NEXT:    [[CROSS_ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 [[CROSS_IDXPROM]], !dbg [[CROSS_LINE7A]]
; CHECK-NEXT:    [[CROSS_LOAD:%.*]] = load i32, ptr [[CROSS_ARRAYIDX]], align 4, !dbg [[CROSS_LINE7A]]
; CHECK-NEXT:    [[CROSS_ADD:%.*]] = add i32 %a, 30, !dbg [[CROSS_LINE6A:![0-9]+]]
; CHECK-NEXT:    [[CROSS_ADD1:%.*]] = add i32 [[CROSS_ADD]], %c, !dbg [[CROSS_LINE6B:![0-9]+]]
; CHECK-NEXT:    [[CROSS_RESULT:%.*]] = add i32 [[CROSS_ADD1]], [[CROSS_LOAD]], !dbg [[CROSS_LINE7B:![0-9]+]]
; CHECK-NEXT:    ret i32 [[CROSS_RESULT]], !dbg [[CROSS_LINE7C:![0-9]+]]
; CHECK:       ![[TMP]] = !DILocalVariable(name: "tmp"
; CHECK:       [[LINE7A]] = !DILocation(line: 7,
; CHECK:       [[LINE6A]] = !DILocation(line: 6,
; CHECK:       [[LINE6B]] = !DILocation(line: 6,
; CHECK:       [[LINE7B]] = !DILocation(line: 7,
; CHECK:       [[LINE7C]] = !DILocation(line: 7,
; CHECK:       ![[CROSS_TMP]] = !DILocalVariable(name: "tmp"
; CHECK:       [[CROSS_LINE7A]] = !DILocation(line: 7,
; CHECK:       [[CROSS_LINE6A]] = !DILocation(line: 6,
; CHECK:       [[CROSS_LINE6B]] = !DILocation(line: 6,
; CHECK:       [[CROSS_LINE7B]] = !DILocation(line: 7,
; CHECK:       [[CROSS_LINE7C]] = !DILocation(line: 7,

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

define i32 @across_basic_blocks(i32 %a, i32 %b, i32 %c) !dbg !18 {
entry:
    #dbg_value(i32 %c, !19, !DIExpression(), !20)
  %add = add nsw i32 %c, 10, !dbg !21
  %add1 = add nsw i32 %add, 20, !dbg !22
    #dbg_value(i32 %add1, !19, !DIExpression(), !20)
  br label %next

next:
  %idxprom = sext i32 %b to i64, !dbg !23
  %arrayidx = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 %idxprom, !dbg !23
  %load = load i32, ptr %arrayidx, align 4, !dbg !23
  %add2 = add nsw i32 %a, %load, !dbg !24
  %result = add nsw i32 %add2, %add1, !dbg !25
  ret i32 %result, !dbg !26
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
!18 = distinct !DISubprogram(name: "across_basic_blocks", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!19 = !DILocalVariable(name: "tmp", scope: !18, file: !1, line: 5, type: !2)
!20 = !DILocation(line: 0, scope: !18)
!21 = !DILocation(line: 6, column: 13, scope: !18)
!22 = !DILocation(line: 6, column: 18, scope: !18)
!23 = !DILocation(line: 7, column: 14, scope: !18)
!24 = !DILocation(line: 7, column: 12, scope: !18)
!25 = !DILocation(line: 7, column: 19, scope: !18)
!26 = !DILocation(line: 7, column: 3, scope: !18)
