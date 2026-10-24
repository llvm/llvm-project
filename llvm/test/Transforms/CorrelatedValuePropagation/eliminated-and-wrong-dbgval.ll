; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @and_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      #dbg_value(i32 %x, [[META18:![0-9]+]], !DIExpression(), [[META19:![0-9]+]])
; CHECK-NEXT: #dbg_value(i32 %x, [[META20:![0-9]+]], !DIExpression(DW_OP_constu, 7, DW_OP_and, DW_OP_stack_value), [[META19]])
; CHECK-NEXT: %cmp = icmp ult i32 %x, 8
;
entry:
    #dbg_value(i32 %x, !14, !DIExpression(), !15)
  %and = and i32 %x, 7, !dbg !16
    #dbg_value(i32 %and, !17, !DIExpression(), !15)
  %cmp = icmp ult i32 %x, 8
  %and. = select i1 %cmp, i32 %and, i32 24
  ret i32 %and.
}

attributes #0 = { noinline nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "and_elide", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!13 = !{}
!14 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 3, type: !12)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocation(line: 4, column: 23, scope: !9)
!17 = !DILocalVariable(name: "masked", scope: !9, file: !1, line: 4, type: !12)
