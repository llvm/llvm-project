; RUN: opt < %s -passes=constraint-elimination -S | FileCheck %s

; Check that checkAndReplaceCondition() salvages the debug value information after replacing
; the conditions (`%t.1` in this test) with the speculated constants (GitHub Issue #135736).
; In particular, debug uses are replaced if the debug record is dominated by the condition fact.

define i1 @test_and_ule(i4 %x, i4 %y, i4 %z) !dbg !5 {
; CHECK-LABEL: define i1 @test_and_ule(
; CHECK-SAME: i4 [[X:%.*]], i4 [[Y:%.*]], i4 [[Z:%.*]])
; CHECK:         [[T_1:%.*]] = icmp ule i4 [[X]], [[Z]], !dbg [[DBG13:![0-9]+]]
;
entry:
  %c.1 = icmp ule i4 %x, %y, !dbg !11
  %c.2 = icmp ule i4 %y, %z, !dbg !12
  %t.1 = icmp ule i4 %x, %z, !dbg !13
  %and = and i1 %c.1, %c.2, !dbg !14
  br i1 %and, label %then, label %exit, !dbg !15

; CHECK:       [[THEN:.*]]:
; CHECK-NEXT:      #dbg_value(i1 true, [[META9:![0-9]+]], !DIExpression(), [[DBG13]])
; CHECK-NEXT:    [[R_1:%.*]] = xor i1 true, true, !dbg [[DBG16:![0-9]+]]
then:                                              ; preds = %entry
    #dbg_value(i1 %t.1, !9, !DIExpression(), !13)
  %r.1 = xor i1 %t.1, %t.1, !dbg !16
  br label %exit

; CHECK:       [[EXIT:.*]]:
; CHECK-NEXT:      #dbg_value(i1 [[T_1]], [[META17:![0-9]+]], !DIExpression(), [[DBG13]])
; CHECK-NEXT:    ret i1 [[T_1]], !dbg [[DBG18:![0-9]+]]
exit:                                             ; preds = %bb1, %entry
    #dbg_value(i1 %t.1, !17, !DIExpression(), !13)
  ret i1 %t.1, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 20}
!3 = !{i32 17}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_and_ule", linkageName: "test_and_ule", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !10)
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 2, column: 1, scope: !5)
!13 = !DILocation(line: 5, column: 1, scope: !5)
!14 = !DILocation(line: 3, column: 1, scope: !5)
!15 = !DILocation(line: 4, column: 1, scope: !5)
!16 = !DILocation(line: 7, column: 1, scope: !5)
!17 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 7, type: !10)
!18 = !DILocation(line: 20, column: 1, scope: !5)

; CHECK: [[META9]] = !DILocalVariable(name: "4",
; CHECK: [[DBG13]] = !DILocation(line: 5, column: 1,
; CHECK: [[DBG16]] = !DILocation(line: 7, column: 1,
; CHECK: [[META17]] = !DILocalVariable(name: "6",
; CHECK: [[DBG18]] = !DILocation(line: 20, column: 1,
;.
