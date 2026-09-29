; RUN: opt -S -passes=licm %s | FileCheck %s

; Check that hoistGEP() in LICM salvages the dbg_value for the hoisted
; getelementptr instruction.

define void @hoist_gep(ptr %ptr, i1 %c, i32 %arg) !dbg !5 {
; CHECK-LABEL: define void @hoist_gep(
; CHECK-LABEL: loop:
; CHECK:           #dbg_value(!DIArgList(ptr [[PTR:%.*]], i64 [[VAL_EXT:%.*]]), [[META9:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), [[META15:![0-9]+]])
;
entry:
  %arg.ext = zext i32 %arg to i64, !dbg !16
  br label %loop, !dbg !17

loop:                                             ; preds = %loop, %entry
  %val = call i32 @get.i32(), !dbg !18
  %val.ext = zext i32 %val to i64, !dbg !19
  %ptr2 = getelementptr inbounds i8, ptr %ptr, i64 %val.ext, !dbg !20
    #dbg_value(ptr %ptr2, !14, !DIExpression(), !20)
  %ptr3 = getelementptr i8, ptr %ptr2, i64 %arg.ext, !dbg !21
  call void @use(ptr %ptr3), !dbg !22
  br i1 %c, label %loop, label %exit, !dbg !23

exit:                                             ; preds = %loop
  ret void, !dbg !24
}

declare i32 @get.i32()
declare void @use(ptr)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "salvage-hoisted-gep.ll", directory: "/")
!2 = !{i32 9}
!3 = !{i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "hoist_gep", linkageName: "hoist_gep", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!14}
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !10)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
!23 = !DILocation(line: 8, column: 1, scope: !5)
!24 = !DILocation(line: 9, column: 1, scope: !5)
;.
; CHECK: [[META9]] = !DILocalVariable(name: "4",
; CHECK: [[META15]] = !DILocation(line: 5,
;.
