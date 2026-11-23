; RUN: opt -S -passes=licm %s | FileCheck %s

; Check that hoistAdd() in LICM salvages the dbg_value for the hoisted add
; instruction.

define i32 @hoist_add(ptr %p, ptr %x_p, ptr %length_p) !dbg !5 {
; CHECK-LABEL: define i32 @hoist_add(
; CHECK-LABEL: loop:
; CHECK:           #dbg_value(!DIArgList(i32 [[X:%.*]], i32 [[IV:%.*]]), [[META9:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), [[META16:![0-9]+]])
;
entry:
  %x = load i32, ptr %x_p, align 4, !dbg !20, !range !21
  %length = load i32, ptr %length_p, align 4, !dbg !22, !range !21
  br label %loop, !dbg !23

loop:                                             ; preds = %backedge, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ], !dbg !24
  %arith = add nsw i32 %x, %iv, !dbg !25
    #dbg_value(i32 %arith, !13, !DIExpression(), !25)
  %x_check = icmp slt i32 %arith, 4, !dbg !26
  br i1 %x_check, label %out_of_bounds, label %backedge, !dbg !27

backedge:                                         ; preds = %loop
  %el.ptr = getelementptr i32, ptr %p, i32 %iv, !dbg !28
  store i32 1, ptr %el.ptr, align 4, !dbg !29
  %iv.next = add nuw nsw i32 %iv, 4, !dbg !30
  %loop_cond = icmp slt i32 %iv.next, %length, !dbg !31
  br i1 %loop_cond, label %loop, label %exit, !dbg !32

exit:                                             ; preds = %backedge
  ret i32 %iv.next, !dbg !33

out_of_bounds:                                    ; preds = %loop
  ret i32 -1, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "salvage-hoisted-add", directory: "/")
!2 = !{i32 14}
!3 = !{i32 8}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "hoist_add", linkageName: "hoist_add", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!13}
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !10)
!20 = !DILocation(line: 1, column: 1, scope: !5)
!21 = !{i32 0, i32 -2147483648}
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
; CHECK: [[META9]] = !DILocalVariable(name: "4",
; CHECK: [[META16]] = !DILocation(line: 5, column: 1,
;.
