; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

define dso_local i32 @udiv_narrow(i32 noundef %x) #0 !dbg !9 {
; CHECK:      entry:
; CHECK-NEXT: %div.lhs.trunc = trunc i32 %x to i8
; CHECK-NEXT: %div1 = udiv i8 %div.lhs.trunc, 42
; CHECK-NEXT: %div.zext = zext i8 %div1 to i32
; CHECK-NEXT: #dbg_value(i32 poison, [[META1:![0-9]+]], !DIExpression(), [[META2:![0-9]+]])
entry:
  %div = udiv i32 %x, 42
    #dbg_value(i32 %div, !17, !DIExpression(), !15)
  %cmp = icmp ult i32 %x, 256
  %div. = select i1 %cmp, i32 %div, i32 24
  ret i32 %div.
}

define dso_local i32 @urem_narrow(i32 noundef %x) #0 !dbg !23 {
; CHECK:      entry:
; CHECK-NEXT: %rem.lhs.trunc = trunc i32 %x to i8
; CHECK-NEXT: %rem1 = urem i8 %rem.lhs.trunc, 42
; CHECK-NEXT: %rem.zext = zext i8 %rem1 to i32
; CHECK-NEXT: #dbg_value(i32 poison, [[META4:![0-9]+]], !DIExpression(), [[META5:![0-9]+]])
entry:
  %rem = urem i32 %x, 42
    #dbg_value(i32 %rem, !27, !DIExpression(), !25)
  %cmp = icmp ult i32 %x, 256
  %rem. = select i1 %cmp, i32 %rem, i32 24
  ret i32 %rem.
}

define dso_local i32 @sdiv_narrow(i32 noundef %x) #0 !dbg !32 {
; CHECK:      entry:
; CHECK-NEXT: %div.lhs.trunc = trunc i32 %x to i8
; CHECK-NEXT: %div1 = sdiv i8 %div.lhs.trunc, 42
; CHECK-NEXT: %div.sext = sext i8 %div1 to i32
; CHECK-NEXT: #dbg_value(i32 %x, [[META7:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_div, DW_OP_stack_value), [[META8:![0-9]+]])
entry:
  %div = sdiv i32 %x, 42
    #dbg_value(i32 %div, !39, !DIExpression(), !37)
  %0 = add i32 %x, 42
  %spec.select = icmp ult i32 %0, 85
  %cond = select i1 %spec.select, i32 %div, i32 24
  ret i32 %cond
}

define dso_local i32 @srem_narrow(i32 noundef %x) #0 !dbg !43 {
; CHECK:      entry:
; CHECK-NEXT: %rem.lhs.trunc = trunc i32 %x to i8
; CHECK-NEXT: %rem1 = srem i8 %rem.lhs.trunc, 42
; CHECK-NEXT: %rem.sext = sext i8 %rem1 to i32
; CHECK-NEXT: #dbg_value(i32 %x, [[META10:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_mod, DW_OP_stack_value), [[META11:![0-9]+]])
entry:
  %rem = srem i32 %x, 42
    #dbg_value(i32 %rem, !47, !DIExpression(), !45)
  %0 = add i32 %x, 42
  %spec.select = icmp ult i32 %0, 85
  %cond = select i1 %spec.select, i32 %rem, i32 24
  ret i32 %cond
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "udiv_narrow", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!13 = !{}
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "quotient", scope: !9, file: !1, line: 4, type: !12)
!23 = distinct !DISubprogram(name: "urem_narrow", scope: !1, file: !1, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!25 = !DILocation(line: 0, scope: !23)
!27 = !DILocalVariable(name: "remainder", scope: !23, file: !1, line: 10, type: !12)
!32 = distinct !DISubprogram(name: "sdiv_narrow", scope: !1, file: !1, line: 15, type: !33, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !35}
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!37 = !DILocation(line: 0, scope: !32)
!39 = !DILocalVariable(name: "quotient", scope: !32, file: !1, line: 16, type: !35)
!43 = distinct !DISubprogram(name: "srem_narrow", scope: !1, file: !1, line: 21, type: !33, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!45 = !DILocation(line: 0, scope: !43)
!47 = !DILocalVariable(name: "remainder", scope: !43, file: !1, line: 22, type: !35)
; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "udiv_narrow"
; CHECK: [[META1]] = !DILocalVariable(name: "quotient", scope: [[META3]]
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
; CHECK: [[META6:![0-9]+]] = distinct !DISubprogram(name: "urem_narrow"
; CHECK: [[META4]] = !DILocalVariable(name: "remainder", scope: [[META6]]
; CHECK: [[META5]] = !DILocation(line: 0, scope: [[META6]])
; CHECK: [[META9:![0-9]+]] = distinct !DISubprogram(name: "sdiv_narrow"
; CHECK: [[META7]] = !DILocalVariable(name: "quotient", scope: [[META9]]
; CHECK: [[META8]] = !DILocation(line: 0, scope: [[META9]])
; CHECK: [[META12:![0-9]+]] = distinct !DISubprogram(name: "srem_narrow"
; CHECK: [[META10]] = !DILocalVariable(name: "remainder", scope: [[META12]]
; CHECK: [[META11]] = !DILocation(line: 0, scope: [[META12]])
