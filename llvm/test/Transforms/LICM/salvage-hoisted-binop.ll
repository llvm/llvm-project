; RUN: opt -S -passes=licm %s | FileCheck %s

; Check that hoistBOAssociation() in LICM salvages the dbg_value for the
; hoisted binary operation.

define void @hoist_binop(i64 %c1, i64 %c2) !dbg !5 {
; CHECK-LABEL: define void @hoist_binop(
; CHECK-LABEL: loop:
; CHECK:           #dbg_value(!DIArgList(i64 [[INDEX:%.*]], i64 [[C1:%.*]]), [[META9:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), [[META13:![0-9]+]])
;
entry:
  br label %loop, !dbg !13

loop:                                             ; preds = %loop, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %loop ], !dbg !14
  %step.add = add i64 %index, %c1, !dbg !15
    #dbg_value(i64 %step.add, !11, !DIExpression(), !15)
  %index.next = add i64 %step.add, %c2, !dbg !16
  br label %loop, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "salvage-hoist-binop.ll", directory: "/")
!2 = !{i32 5}
!3 = !{i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "hoist_binop", linkageName: "hoist_binop", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!11}
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !10)
!13 = !DILocation(line: 1, column: 1, scope: !5)
!14 = !DILocation(line: 2, column: 1, scope: !5)
!15 = !DILocation(line: 3, column: 1, scope: !5)
!16 = !DILocation(line: 4, column: 1, scope: !5)
!17 = !DILocation(line: 5, column: 1, scope: !5)
;.
; CHECK: [[META9]] = !DILocalVariable(name: "2",
; CHECK: [[META13]] = !DILocation(line: 3, column: 1,
;.
