; RUN: opt < %s -passes=constraint-elimination -S | FileCheck %s

; Check that checkAndReplaceCondition() salvages the debug value information after replacing
; the conditions (`%c.1` and `%t.2` in this test) with the speculated constants (GitHub Issue
; #135736).
; In particular, the debug value record uses should not be replaced if they come before the
; context instrtuction (e.g., `%t.2` in this example).

declare void @llvm.assume(i1 noundef) #0

declare void @may_unwind()

declare void @use(i1)

define i1 @assume_single_bb_conditions_after_assume(i8 %a, i8 %b, i1 %c) !dbg !5 {
; CHECK-LABEL: define i1 @assume_single_bb_conditions_after_assume(
; CHECK:         [[CMP_1:%.*]] = icmp ule i8 [[ADD_1:%.*]], [[B:%.*]], !dbg [[DBG12:![0-9]+]]
; CHECK-NEXT:         [[C_1:%.*]] = icmp ule i8 [[ADD_1]], [[B]], !dbg [[DBG13:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i1 [[C_1]], [[META9:![0-9]+]], !DIExpression(), [[META14:![0-9]+]])
; CHECK-NEXT:    call void @use(i1 [[C_1]]), !dbg [[DBG15:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i1 [[C_1]], [[META9]], !DIExpression(), [[META14]])
; CHECK-NEXT:    call void @may_unwind(), !dbg [[DBG16:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i1 [[C_1]], [[META9]], !DIExpression(), [[META14]])
; CHECK-NEXT:    call void @llvm.assume(i1 [[CMP_1]]), !dbg [[DBG17:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i1 [[C_1]], [[META9]], !DIExpression(), [[META14]])
; CHECK-NEXT:      #dbg_value(i1 true, [[META9]], !DIExpression(), [[META14]])
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 true, true, !dbg [[DBG18:![0-9]+]]
;
  %add.1 = add nuw nsw i8 %a, 1, !dbg !11
  %cmp.1 = icmp ule i8 %add.1, %b, !dbg !12
  %c.1 = icmp ule i8 %add.1, %b, !dbg !13
    #dbg_value(i1 %c.1, !9, !DIExpression(), !14)
  call void @use(i1 %c.1), !dbg !15
    #dbg_value(i1 %c.1, !9, !DIExpression(), !14)
  call void @may_unwind(), !dbg !16
    #dbg_value(i1 %c.1, !9, !DIExpression(), !14)
  call void @llvm.assume(i1 %cmp.1), !dbg !17
    #dbg_value(i1 %c.1, !9, !DIExpression(), !14)
  %t.2 = icmp ule i8 %a, %b, !dbg !14
    #dbg_value(i1 %c.1, !9, !DIExpression(), !14)
  %res.1 = xor i1 %c.1, %t.2, !dbg !18
  %add.2 = add nuw nsw i8 %a, 2, !dbg !19
  %c.2 = icmp ule i8 %add.2, %b, !dbg !20
  %res.2 = xor i1 %res.1, %c.2, !dbg !21
  ret i1 %res.2, !dbg !22
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/app/example.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 8}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "assume_single_bb_conditions_after_assume", linkageName: "assume_single_bb_conditions_after_assume", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 7, type: !10)
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 2, column: 1, scope: !5)
!13 = !DILocation(line: 3, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 4, column: 1, scope: !5)
!16 = !DILocation(line: 5, column: 1, scope: !5)
!17 = !DILocation(line: 6, column: 1, scope: !5)
!18 = !DILocation(line: 8, column: 1, scope: !5)
!19 = !DILocation(line: 9, column: 1, scope: !5)
!20 = !DILocation(line: 10, column: 1, scope: !5)
!21 = !DILocation(line: 11, column: 1, scope: !5)
!22 = !DILocation(line: 12, column: 1, scope: !5)

; CHECK: [[META9]] = !DILocalVariable(name: "4",
; CHECK: [[DBG12]] = !DILocation(line: 2, column: 1,
; CHECK: [[DBG13]] = !DILocation(line: 3, column: 1,
; CHECK: [[META14]] = !DILocation(line: 7, column: 1,
; CHECK: [[DBG18]] = !DILocation(line: 8, column: 1,
;.
