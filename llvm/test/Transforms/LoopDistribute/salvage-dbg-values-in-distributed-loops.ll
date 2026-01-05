; RUN: opt -passes=loop-distribute -enable-loop-distribute -S < %s  | FileCheck %s

; Check that removeUnusedInsts() salvages `dbg_value`s which use dead
; instructions in the distributed loops.

define void @f(ptr noalias %a, ptr noalias %c, ptr noalias %d) !dbg !5 {
; CHECK-LABEL: define void @f(
; CHECK-SAME: ptr noalias [[A:%.*]], ptr noalias [[C:%.*]], ptr noalias [[D:%.*]])

entry:
  br label %for.body, !dbg !21

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ], !dbg !22
  %add = add nuw nsw i64 %ind, 1, !dbg !23

; CHECK-LABEL: for.body.ldist1:
; CHECK: #dbg_value(!DIArgList(ptr [[D]], i64 [[IND_LDIST1:%.*]]), [[META16:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[META28:![0-9]+]])
;
  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind, !dbg !24
    #dbg_value(ptr %arrayidxA, !12, !DIExpression(), !24)
  %loadA = load i32, ptr %arrayidxA, align 4, !dbg !25
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add, !dbg !26
  store i32 %loadA, ptr %arrayidxA_plus_4, align 4, !dbg !27

; CHECK-LABEL: for.body:
; CHECK: #dbg_value(!DIArgList(ptr [[A]], i64 [[IND:%.*]]), [[META12:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), [[DBG24:![0-9]+]])
;
  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind, !dbg !28
    #dbg_value(ptr %arrayidxD, !16, !DIExpression(), !28)
  %loadD = load i32, ptr %arrayidxD, align 4, !dbg !29
  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind, !dbg !30
  store i32 %loadD, ptr %arrayidxC, align 4, !dbg !31

  %exitcond = icmp eq i64 %add, 20, !dbg !32
  br i1 %exitcond, label %for.end, label %for.body, !dbg !33

for.end:                                          ; preds = %for.body
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 14}
!3 = !{i32 9}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!12, !16}
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !10)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 8, type: !10)
!21 = !DILocation(line: 1, column: 1, scope: !5)
!22 = !DILocation(line: 2, column: 1, scope: !5)
!23 = !DILocation(line: 3, column: 1, scope: !5)
!24 = !DILocation(line: 4, column: 1, scope: !5)
!25 = !DILocation(line: 5, column: 1, scope: !5)
!26 = !DILocation(line: 6, column: 1, scope: !5)
!27 = !DILocation(line: 7, column: 1, scope: !5)
!28 = !DILocation(line: 8, column: 1, scope: !5)
!29 = !DILocation(line: 9, column: 1, scope: !5)
!30 = !DILocation(line: 10, column: 1, scope: !5)
!31 = !DILocation(line: 11, column: 1, scope: !5)
!32 = !DILocation(line: 12, column: 1, scope: !5)
!33 = !DILocation(line: 13, column: 1, scope: !5)
!34 = !DILocation(line: 14, column: 1, scope: !5)
;.
; CHECK: [[META12]] = !DILocalVariable(name: "3"
; CHECK: [[META16]] = !DILocalVariable(name: "6"
; CHECK: [[DBG24]] = !DILocation(line: 4, column: 1
; CHECK: [[META28]] = !DILocation(line: 8, column: 1
;.
