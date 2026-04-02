; RUN: opt -passes=newgvn -S %s | FileCheck %s

; Check that assignDFSNumbers() in NewGVN salvages the debug values of the
; trivially dead instructions that are marked for deletion.

; CHECK: #dbg_value(i8 %tmp, [[META11:![0-9]+]], !DIExpression(DW_OP_constu, 8, DW_OP_eq, DW_OP_stack_value), [[META26:![0-9]+]])
; CHECK: [[META11]] = !DILocalVariable(name: "2"
; CHECK: [[META26]] = !DILocation(line: 2

define void @test13() !dbg !5 {
entry:
  %tmp = load i8, ptr null, align 1
  %tmp2 = icmp eq i8 %tmp, 8, !dbg !13
    #dbg_value(i1 %tmp2, !11, !DIExpression(), !13)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/app/example.ll", directory: "/")
!2 = !{i32 3}
!3 = !{i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test13", linkageName: "test13", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!11}
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 2, column: 1, scope: !5)