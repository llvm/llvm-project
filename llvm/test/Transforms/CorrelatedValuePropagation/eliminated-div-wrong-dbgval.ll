; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When removing div instructions, any debug values need to be updated correctly
; to the value they would have had (or be poisoned)
; See issues #218630 & #218632

define dso_local i32 @sdiv_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      entry:
; CHECK-NEXT: #dbg_value(i32 poison, [[META1:![0-9]+]], !DIExpression(), [[META2:![0-9]+]])
; CHECK-NEXT: %0 = add i32 %x, 41
entry:
  %div = sdiv i32 %x, 42
    #dbg_value(i32 %div, !17, !DIExpression(), !15)
  %0 = add i32 %x, 41
  %spec.select = icmp ult i32 %0, 83
  %cond = select i1 %spec.select, i32 %div, i32 24
  ret i32 %cond
}

define dso_local i32 @udiv_elide(i32 noundef %x) #0 !dbg !23 {
; CHECK:      entry:
; CHECK-NEXT: #dbg_value(i32 %x, [[META4:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_div, DW_OP_stack_value), [[META5:![0-9]+]])
; CHECK-NEXT: %cmp = icmp ult i32 %x, 42
entry:
  %div = udiv i32 %x, 42
    #dbg_value(i32 %div, !30, !DIExpression(), !28)
  %cmp = icmp ult i32 %x, 42
  %div. = select i1 %cmp, i32 %div, i32 24
  ret i32 %div.
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "sdiv_elide", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "quotient", scope: !9, file: !1, line: 4, type: !12)
!23 = distinct !DISubprogram(name: "udiv_elide", scope: !1, file: !1, line: 9, type: !24, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!24 = !DISubroutineType(types: !25)
!25 = !{!26, !26}
!26 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!28 = !DILocation(line: 0, scope: !23)
!30 = !DILocalVariable(name: "quotient", scope: !23, file: !1, line: 10, type: !26)
; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "sdiv_elide"
; CHECK: [[META1]] = !DILocalVariable(name: "quotient", scope: [[META3]]
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
; CHECK: [[META6:![0-9]+]] = distinct !DISubprogram(name: "udiv_elide"
; CHECK: [[META4]] = !DILocalVariable(name: "quotient", scope: [[META6]]
; CHECK: [[META5]] = !DILocation(line: 0, scope: [[META6]])
