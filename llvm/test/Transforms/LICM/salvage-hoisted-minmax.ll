; RUN: opt -S -passes=licm %s | FileCheck %s

; Check that hoistMinMax() in LICM salvages the debug values for the hoisted
; cmp instructions.

define i32 @test_ult(i32 %start, i32 %inv_1, i32 %inv_2) !dbg !5 {
; CHECK-LABEL: define i32 @test_ult(
; CHECK-SAME: i32 [[START:%.*]], i32 [[INV_1:%.*]], i32 [[INV_2:%.*]])
; CHECK:       [[LOOP:.*]]:
; CHECK:           #dbg_value(!DIArgList(i32 [[IV:%.*]], i32 [[INV_1]]), [[META9:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_lt, DW_OP_stack_value), [[META14:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(i32 [[IV]], i32 [[INV_2]]), [[META11:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_lt, DW_OP_stack_value), [[META15:![0-9]+]])
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[IV]], [[INVARIANT_UMIN:%.*]], !dbg [[DBG16:![0-9]+]]
entry:
  br label %loop, !dbg !16

loop:                                             ; preds = %loop, %entry
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ], !dbg !17
  %cmp_1 = icmp ult i32 %iv, %inv_1, !dbg !18
    #dbg_value(i1 %cmp_1, !11, !DIExpression(), !18)
  %cmp_2 = icmp ult i32 %iv, %inv_2, !dbg !19
    #dbg_value(i1 %cmp_2, !13, !DIExpression(), !19)
  %loop_cond = and i1 %cmp_1, %cmp_2, !dbg !20
  %iv.next = add i32 %iv, 1, !dbg !21
  br i1 %loop_cond, label %loop, label %exit, !dbg !22

exit:                                             ; preds = %loop
  ret i32 %iv, !dbg !23
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 8}
!3 = !{i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_ult", linkageName: "test_ult", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!11, !13}
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !12)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
!23 = !DILocation(line: 8, column: 1, scope: !5)
;.
; CHECK: [[META9]] = !DILocalVariable(name: "2",
; CHECK: [[META11]] = !DILocalVariable(name: "3",
; CHECK: [[META14]] = !DILocation(line: 3,
; CHECK: [[META15]] = !DILocation(line: 4,
;.
