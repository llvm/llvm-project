; RUN: opt < %s -passes=reassociate -S | FileCheck %s

@g = internal constant [2 x i32] [i32 9, i32 4], align 4

; Reduced from issue #212714. Reassociation moves the line 6 expression after
; part of line 7. Preserve the salvaged tmp update at its original program point,
; restore tmp's old value for the moved expression, then reapply the update.

define i32 @example(i32 %a, i32 %b, i32 %c) !dbg !5 {
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
;
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

; Reassociation can also move the expression from a predecessor into the root's
; block. Do not add block-local variable view transitions in that case.

define i32 @across_basic_blocks(i32 %a, i32 %b, i32 %c) !dbg !18 {
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
;
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

; If the predecessor can exit along another path, moving its expression into a
; successor changes the paths on which it executes. Drop the moved locations.

define i32 @across_conditional_edge(i32 %a, i32 %b, i32 %c) !dbg !27 {
; CHECK-LABEL: @across_conditional_edge(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[COND:%.*]] = icmp ne i32 %b, 0
; CHECK-NEXT:    br i1 [[COND]], label %next, label %exit
; CHECK:       next:
; CHECK-NEXT:    [[COND_IDXPROM:%.*]] = sext i32 %b to i64, !dbg [[COND_LINE7A:![0-9]+]]
; CHECK-NEXT:    [[COND_ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 [[COND_IDXPROM]], !dbg [[COND_LINE7A]]
; CHECK-NEXT:    [[COND_LOAD:%.*]] = load i32, ptr [[COND_ARRAYIDX]], align 4, !dbg [[COND_LINE7A]]
; CHECK-NEXT:    [[COND_ADD:%.*]] = add i32 %a, 30{{$}}
; CHECK-NEXT:    [[COND_ADD1:%.*]] = add i32 [[COND_ADD]], %c{{$}}
; CHECK-NEXT:    [[COND_RESULT:%.*]] = add i32 [[COND_ADD1]], [[COND_LOAD]], !dbg [[COND_LINE7B:![0-9]+]]
; CHECK-NEXT:    ret i32 [[COND_RESULT]], !dbg [[COND_LINE7C:![0-9]+]]
; CHECK:       exit:
; CHECK-NEXT:    ret i32 0, !dbg [[COND_LINE8:![0-9]+]]
;
entry:
  %add = add nsw i32 %c, 10, !dbg !28
  %add1 = add nsw i32 %add, 20, !dbg !29
  %cond = icmp ne i32 %b, 0
  br i1 %cond, label %next, label %exit

next:
  %idxprom = sext i32 %b to i64, !dbg !30
  %arrayidx = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 %idxprom, !dbg !30
  %load = load i32, ptr %arrayidx, align 4, !dbg !30
  %add2 = add nsw i32 %a, %load, !dbg !31
  %result = add nsw i32 %add2, %add1, !dbg !32
  ret i32 %result, !dbg !33

exit:
  ret i32 0, !dbg !34
}

; A single CFG successor is not sufficient if execution can exit implicitly
; between the old and new positions.

define i32 @across_implicit_exit(i32 %a, i32 %b, i32 %c) !dbg !35 {
; CHECK-LABEL: @across_implicit_exit(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @may_not_return()
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    [[EXIT_IDXPROM:%.*]] = sext i32 %b to i64, !dbg [[EXIT_LINE7A:![0-9]+]]
; CHECK-NEXT:    [[EXIT_ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 [[EXIT_IDXPROM]], !dbg [[EXIT_LINE7A]]
; CHECK-NEXT:    [[EXIT_LOAD:%.*]] = load i32, ptr [[EXIT_ARRAYIDX]], align 4, !dbg [[EXIT_LINE7A]]
; CHECK-NEXT:    [[EXIT_ADD:%.*]] = add i32 %a, 30{{$}}
; CHECK-NEXT:    [[EXIT_ADD1:%.*]] = add i32 [[EXIT_ADD]], %c{{$}}
; CHECK-NEXT:    [[EXIT_RESULT:%.*]] = add i32 [[EXIT_ADD1]], [[EXIT_LOAD]], !dbg [[EXIT_LINE7B:![0-9]+]]
; CHECK-NEXT:    ret i32 [[EXIT_RESULT]], !dbg [[EXIT_LINE7C:![0-9]+]]
;
entry:
  %add = add nsw i32 %c, 10, !dbg !36
  %add1 = add nsw i32 %add, 20, !dbg !37
  call void @may_not_return()
  br label %next

next:
  %idxprom = sext i32 %b to i64, !dbg !38
  %arrayidx = getelementptr inbounds [2 x i32], ptr @g, i64 0, i64 %idxprom, !dbg !38
  %load = load i32, ptr %arrayidx, align 4, !dbg !38
  %add2 = add nsw i32 %a, %load, !dbg !39
  %result = add nsw i32 %add2, %add1, !dbg !40
  ret i32 %result, !dbg !41
}

declare void @may_not_return()

; CHECK: ![[TMP]] = !DILocalVariable(name: "tmp"
; CHECK: [[LINE7A]] = !DILocation(line: 7,
; CHECK: [[LINE6A]] = !DILocation(line: 6,
; CHECK: [[LINE6B]] = !DILocation(line: 6,
; CHECK: [[LINE7B]] = !DILocation(line: 7,
; CHECK: [[LINE7C]] = !DILocation(line: 7,
; CHECK: ![[CROSS_TMP]] = !DILocalVariable(name: "tmp"
; CHECK: [[CROSS_LINE7A]] = !DILocation(line: 7,
; CHECK: [[CROSS_LINE6A]] = !DILocation(line: 6,
; CHECK: [[CROSS_LINE6B]] = !DILocation(line: 6,
; CHECK: [[CROSS_LINE7B]] = !DILocation(line: 7,
; CHECK: [[CROSS_LINE7C]] = !DILocation(line: 7,
; CHECK: [[COND_LINE7A]] = !DILocation(line: 7,
; CHECK: [[COND_LINE7B]] = !DILocation(line: 7,
; CHECK: [[COND_LINE7C]] = !DILocation(line: 7,
; CHECK: [[COND_LINE8]] = !DILocation(line: 8,
; CHECK: [[EXIT_LINE7A]] = !DILocation(line: 7,
; CHECK: [[EXIT_LINE7B]] = !DILocation(line: 7,
; CHECK: [[EXIT_LINE7C]] = !DILocation(line: 7,

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
!27 = distinct !DISubprogram(name: "across_conditional_edge", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!28 = !DILocation(line: 6, column: 13, scope: !27)
!29 = !DILocation(line: 6, column: 18, scope: !27)
!30 = !DILocation(line: 7, column: 14, scope: !27)
!31 = !DILocation(line: 7, column: 12, scope: !27)
!32 = !DILocation(line: 7, column: 19, scope: !27)
!33 = !DILocation(line: 7, column: 3, scope: !27)
!34 = !DILocation(line: 8, column: 3, scope: !27)
!35 = distinct !DISubprogram(name: "across_implicit_exit", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!36 = !DILocation(line: 6, column: 13, scope: !35)
!37 = !DILocation(line: 6, column: 18, scope: !35)
!38 = !DILocation(line: 7, column: 14, scope: !35)
!39 = !DILocation(line: 7, column: 12, scope: !35)
!40 = !DILocation(line: 7, column: 19, scope: !35)
!41 = !DILocation(line: 7, column: 3, scope: !35)
