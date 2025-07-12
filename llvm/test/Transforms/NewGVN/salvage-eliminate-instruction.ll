; RUN: opt -S -passes=newgvn %s | FileCheck %s

; Check that eliminateInstruction() salvages the debug value of `Def` (`DefI`)
; which is marked for deletion.


define void @binop(i32 %x, i32 %y) !dbg !5 {
; CHECK: #dbg_value(!DIArgList(i32 %y, i32 %x), [[META11:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), [[META13:![0-9]+]])
;
  %add1 = add i32 %x, %y, !dbg !12
    #dbg_value(i32 %add1, !9, !DIExpression(), !12)
  %add2 = add i32 %y, %x, !dbg !13
    #dbg_value(i32 %add2, !11, !DIExpression(), !13)
  call void @use(i32 %add1, i32 %add2), !dbg !14
  ret void, !dbg !15
}

declare void @use(i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/app/example.ll", directory: "/")
!2 = !{i32 4}
!3 = !{i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "binop", linkageName: "binop", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 1, column: 1, scope: !5)
!13 = !DILocation(line: 2, column: 1, scope: !5)
!14 = !DILocation(line: 3, column: 1, scope: !5)
!15 = !DILocation(line: 4, column: 1, scope: !5)
;.
; CHECK: [[META11]] = !DILocalVariable(name: "2",
; CHECK: [[META13]] = !DILocation(line: 2,
;.
