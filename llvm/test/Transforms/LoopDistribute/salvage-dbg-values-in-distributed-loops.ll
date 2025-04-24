; RUN: opt -passes=loop-distribute -enable-loop-distribute -S < %s  | FileCheck %s

define void @f(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %d, ptr noalias %e) !dbg !5 {
; CHECK-LABEL: define void @f(
; CHECK-SAME: ptr noalias [[A:%.*]], ptr noalias [[B:%.*]], ptr noalias [[C:%.*]], ptr noalias [[D:%.*]], ptr noalias [[E:%.*]])

; CHECK:       for.body.ldist1:
; CHECK-NEXT:    [[IND_LDIST1:%.*]] = phi i64 [ 0, %[[ENTRY_SPLIT_LDIST1:.*]] ], [ [[ADD_LDIST1:%.*]], %for.body.ldist1 ], !dbg [[DBG28:![0-9]+]]

; CHECK:           store i32
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[D]], i64 [[IND_LDIST1]]), [[META19:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[META37:![0-9]+]])
; CHECK-NEXT:      #dbg_value(i32 poison, [[META20:![0-9]+]], !DIExpression(), [[META38:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[E]], i64 [[IND_LDIST1]]), [[META21:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[META39:![0-9]+]])
; CHECK-NEXT:      #dbg_value(i32 poison, [[META22:![0-9]+]], !DIExpression(), [[META40:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(i32 poison, i32 poison), [[META23:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value), [[META41:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[C]], i64 [[IND_LDIST1]]), [[META24:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[META42:![0-9]+]])
; CHECK-NEXT:    [[EXITCOND_LDIST1:%.*]] = icmp eq i64 [[ADD_LDIST1]], 20, !dbg [[DBG43:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i1 [[EXITCOND_LDIST1]], [[META25:![0-9]+]], !DIExpression(), [[DBG43]])
; CHECK-NEXT:    br i1 [[EXITCOND_LDIST1]], label %[[ENTRY_SPLIT:.*]], label %for.body.ldist1, !dbg [[DBG44:![0-9]+]]

; CHECK:       for.body:
; CHECK-NEXT:    [[IND:%.*]] = phi i64 [ 0, %[[ENTRY_SPLIT:.*]] ], [ [[ADD:%.*]], %for.body ], !dbg [[DBG28]]
; CHECK-NEXT:      #dbg_value(i64 [[IND]], [[META9:![0-9]+]], !DIExpression(), [[DBG28]])
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[A]], i64 [[IND]]), [[META11:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[DBG29:![0-9]+]])
; CHECK-NEXT:      #dbg_value(i32 poison, [[META12:![0-9]+]], !DIExpression(), [[DBG30:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[B]], i64 [[IND]]), [[META14:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[DBG31:![0-9]+]])
; CHECK-NEXT:      #dbg_value(i32 poison, [[META15:![0-9]+]], !DIExpression(), [[DBG32:![0-9]+]])
; CHECK-NEXT:      #dbg_value(!DIArgList(i32 poison, i32 poison), [[META16:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value), [[DBG33:![0-9]+]])
; CHECK-NEXT:    [[ADD]] = add nuw nsw i64 [[IND]], 1, !dbg [[DBG34:![0-9]+]]
; CHECK-NEXT:      #dbg_value(i64 [[ADD]], [[META17:![0-9]+]], !DIExpression(), [[DBG34]])
; CHECK-NEXT:      #dbg_value(!DIArgList(ptr [[A]], i64 [[ADD]]), [[META18:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[DBG35:![0-9]+]])
;
entry:
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ], !dbg !28
  #dbg_value(i64 %ind, !9, !DIExpression(), !28)
  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind, !dbg !29
  #dbg_value(ptr %arrayidxA, !11, !DIExpression(), !29)
  %loadA = load i32, ptr %arrayidxA, align 4, !dbg !30
  #dbg_value(i32 %loadA, !12, !DIExpression(), !30)
  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind, !dbg !31
  #dbg_value(ptr %arrayidxB, !14, !DIExpression(), !31)
  %loadB = load i32, ptr %arrayidxB, align 4, !dbg !32
  #dbg_value(i32 %loadB, !15, !DIExpression(), !32)
  %mulA = mul i32 %loadB, %loadA, !dbg !33
  #dbg_value(i32 %mulA, !16, !DIExpression(), !33)
  %add = add nuw nsw i64 %ind, 1, !dbg !34
  #dbg_value(i64 %add, !17, !DIExpression(), !34)
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add, !dbg !35
  #dbg_value(ptr %arrayidxA_plus_4, !18, !DIExpression(), !35)
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4, !dbg !36
  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind, !dbg !37
  #dbg_value(ptr %arrayidxD, !19, !DIExpression(), !37)
  %loadD = load i32, ptr %arrayidxD, align 4, !dbg !38
  #dbg_value(i32 %loadD, !20, !DIExpression(), !38)
  %arrayidxE = getelementptr inbounds i32, ptr %e, i64 %ind, !dbg !39
  #dbg_value(ptr %arrayidxE, !21, !DIExpression(), !39)
  %loadE = load i32, ptr %arrayidxE, align 4, !dbg !40
  #dbg_value(i32 %loadE, !22, !DIExpression(), !40)
  %mulC = mul i32 %loadD, %loadE, !dbg !41
  #dbg_value(i32 %mulC, !23, !DIExpression(), !41)
  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind, !dbg !42
  #dbg_value(ptr %arrayidxC, !24, !DIExpression(), !42)
  store i32 %mulC, ptr %arrayidxC, align 4, !dbg !43
  %exitcond = icmp eq i64 %add, 20, !dbg !44
  #dbg_value(i1 %exitcond, !25, !DIExpression(), !44)
  br i1 %exitcond, label %for.end, label %for.body, !dbg !45

for.end:                                          ; preds = %for.body
  ret void, !dbg !46
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/app/example.ll", directory: "/")
!2 = !{i32 20}
!3 = !{i32 15}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !12, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !10)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !13)
!13 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !10)
!15 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 6, type: !13)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 7, type: !13)
!17 = !DILocalVariable(name: "7", scope: !5, file: !1, line: 8, type: !10)
!18 = !DILocalVariable(name: "8", scope: !5, file: !1, line: 9, type: !10)
!19 = !DILocalVariable(name: "9", scope: !5, file: !1, line: 11, type: !10)
!20 = !DILocalVariable(name: "10", scope: !5, file: !1, line: 12, type: !13)
!21 = !DILocalVariable(name: "11", scope: !5, file: !1, line: 13, type: !10)
!22 = !DILocalVariable(name: "12", scope: !5, file: !1, line: 14, type: !13)
!23 = !DILocalVariable(name: "13", scope: !5, file: !1, line: 15, type: !13)
!24 = !DILocalVariable(name: "14", scope: !5, file: !1, line: 16, type: !10)
!25 = !DILocalVariable(name: "15", scope: !5, file: !1, line: 18, type: !26)
!26 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!27 = !DILocation(line: 1, column: 1, scope: !5)
!28 = !DILocation(line: 2, column: 1, scope: !5)
!29 = !DILocation(line: 3, column: 1, scope: !5)
!30 = !DILocation(line: 4, column: 1, scope: !5)
!31 = !DILocation(line: 5, column: 1, scope: !5)
!32 = !DILocation(line: 6, column: 1, scope: !5)
!33 = !DILocation(line: 7, column: 1, scope: !5)
!34 = !DILocation(line: 8, column: 1, scope: !5)
!35 = !DILocation(line: 9, column: 1, scope: !5)
!36 = !DILocation(line: 10, column: 1, scope: !5)
!37 = !DILocation(line: 11, column: 1, scope: !5)
!38 = !DILocation(line: 12, column: 1, scope: !5)
!39 = !DILocation(line: 13, column: 1, scope: !5)
!40 = !DILocation(line: 14, column: 1, scope: !5)
!41 = !DILocation(line: 15, column: 1, scope: !5)
!42 = !DILocation(line: 16, column: 1, scope: !5)
!43 = !DILocation(line: 17, column: 1, scope: !5)
!44 = !DILocation(line: 18, column: 1, scope: !5)
!45 = !DILocation(line: 19, column: 1, scope: !5)
!46 = !DILocation(line: 20, column: 1, scope: !5)
;.
; CHECK: [[META9]] = !DILocalVariable(name: "1",
; CHECK: [[META11]] = !DILocalVariable(name: "2",
; CHECK: [[META12]] = !DILocalVariable(name: "3",
; CHECK: [[META14]] = !DILocalVariable(name: "4",
; CHECK: [[META15]] = !DILocalVariable(name: "5",
; CHECK: [[META16]] = !DILocalVariable(name: "6",
; CHECK: [[META17]] = !DILocalVariable(name: "7",
; CHECK: [[META18]] = !DILocalVariable(name: "8",
; CHECK: [[META19]] = !DILocalVariable(name: "9",
; CHECK: [[META20]] = !DILocalVariable(name: "10",
; CHECK: [[META21]] = !DILocalVariable(name: "11",
; CHECK: [[META22]] = !DILocalVariable(name: "12",
; CHECK: [[META23]] = !DILocalVariable(name: "13",
; CHECK: [[META24]] = !DILocalVariable(name: "14",
; CHECK: [[META25]] = !DILocalVariable(name: "15",
; CHECK: [[DBG28]] = !DILocation(line: 2, column: 1,
; CHECK: [[DBG29]] = !DILocation(line: 3, column: 1,
; CHECK: [[DBG30]] = !DILocation(line: 4, column: 1,
; CHECK: [[DBG31]] = !DILocation(line: 5, column: 1,
; CHECK: [[DBG33]] = !DILocation(line: 7, column: 1,
; CHECK: [[DBG34]] = !DILocation(line: 8, column: 1,
; CHECK: [[DBG35]] = !DILocation(line: 9, column: 1,
; CHECK: [[META37]] = !DILocation(line: 11, column: 1,
; CHECK: [[META39]] = !DILocation(line: 13, column: 1,
; CHECK: [[META40]] = !DILocation(line: 14, column: 1,
; CHECK: [[META41]] = !DILocation(line: 15, column: 1,
; CHECK: [[META42]] = !DILocation(line: 16, column: 1,
; CHECK: [[DBG43]] = !DILocation(line: 18, column: 1,
; CHECK: [[DBG44]] = !DILocation(line: 19, column: 1,
;.
