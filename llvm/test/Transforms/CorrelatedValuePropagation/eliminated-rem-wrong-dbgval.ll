; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When removing srem/urem instructions, any debug values need to be updated correctly
; to the value they would have had (or be poisoned)
; See issues #218631 and #218633

define dso_local i32 @srem_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      #dbg_value(i32 poison, [[META2:![0-9]+]], !DIExpression(), [[META1:![0-9]+]])
; CHECK-NEXT: %0 = add i32 %x, 41
;
entry:
  %rem1 = srem i32 %x, 42
    #dbg_value(i32 %rem1, !17, !DIExpression(), !15)
  %0 = add i32 %x, 41
  %spec.select = icmp ult i32 %0, 83
  %cond = select i1 %spec.select, i32 %rem1, i32 24
  ret i32 %cond
}

define dso_local i32 @urem_elide(i32 noundef %x) #0 !dbg !23 {
; CHECK:      #dbg_value(i32 %x, [[META4:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_mod, DW_OP_stack_value), [[META3:![0-9]+]])
; CHECK-NEXT: %cmp = icmp ult i32 %x, 42
;
entry:
  %rem = urem i32 %x, 42
    #dbg_value(i32 %rem, !30, !DIExpression(), !28)
  %cmp = icmp ult i32 %x, 42
  %rem. = select i1 %cmp, i32 %rem, i32 24
  ret i32 %rem.
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "srem_elide", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "rem", scope: !9, file: !1, line: 4, type: !12)
!23 = distinct !DISubprogram(name: "urem_elide", scope: !1, file: !1, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!24 = !DISubroutineType(types: !25)
!25 = !{!26, !26}
!26 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!28 = !DILocation(line: 0, scope: !23)
!30 = !DILocalVariable(name: "remainder", scope: !23, file: !1, line: 9, type: !26)

; CHECK: [[META5:![0-9]+]] = distinct !DISubprogram(name: "srem_elide"
; CHECK: [[META2]] = !DILocalVariable(name: "rem", scope: [[META5]]
; CHECK: [[META1]] = !DILocation(line: 0, scope: [[META5]])
; CHECK: [[META6:![0-9]+]] = distinct !DISubprogram(name: "urem_elide"
; CHECK: [[META4]] = !DILocalVariable(name: "remainder", scope: [[META6]]
; CHECK: [[META3]] = !DILocation(line: 0, scope: [[META6]])
