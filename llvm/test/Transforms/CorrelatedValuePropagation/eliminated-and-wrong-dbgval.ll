; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When removing and instructions, any debug values need to be updated correctly
; to the value they would have had (or be poisoned)
; See issues #219940

define dso_local i32 @and_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      #dbg_value(i32 %x, [[META1:![0-9]+]], !DIExpression(DW_OP_constu, 7, DW_OP_and, DW_OP_stack_value), [[META2:![0-9]+]])
; CHECK-NEXT: %cmp = icmp ult i32 %x, 8
;
entry:
  %and = and i32 %x, 7
    #dbg_value(i32 %and, !17, !DIExpression(), !15)
  %cmp = icmp ult i32 %x, 8
  %and. = select i1 %cmp, i32 %and, i32 24
  ret i32 %and.
}

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
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "masked", scope: !9, file: !1, line: 4, type: !12)

; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "and_elide"
; CHECK: [[META1]] = !DILocalVariable(name: "masked", scope: [[META3]] 
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
