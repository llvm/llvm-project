; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @srem_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      #dbg_value(i32 %x, [[META14:![0-9]+]], !DIExpression(), [[META15:![0-9]+]])
; CHECK-NEXT: #dbg_value(i32 %x, [[META16:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_mod, DW_OP_stack_value), [[META15]])
; CHECK-NEXT: %0 = add i32 %x, 41
;
entry:
    #dbg_value(i32 %x, !14, !DIExpression(), !15)
  %rem1 = srem i32 %x, 42, !dbg !16
    #dbg_value(i32 %rem1, !17, !DIExpression(), !15)
  %0 = add i32 %x, 41
  %spec.select = icmp ult i32 %0, 83
  %cond = select i1 %spec.select, i32 %rem1, i32 24
  ret i32 %cond
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @urem_elide(i32 noundef %x) #0 !dbg !23 {
; CHECK:      #dbg_value(i32 %x, [[META26:![0-9]+]], !DIExpression(), [[META27:![0-9]+]])
; CHECK-NEXT: #dbg_value(i32 poison, [[META28:![0-9]+]], !DIExpression(), [[META27]])
; CHECK-NEXT: %cmp = icmp ult i32 %x, 42
;
entry:
    #dbg_value(i32 %x, !27, !DIExpression(), !28)
  %rem = urem i32 %x, 42, !dbg !29
    #dbg_value(i32 %rem, !30, !DIExpression(), !28)
  %cmp = icmp ult i32 %x, 42
  %rem. = select i1 %cmp, i32 %rem, i32 24
  ret i32 %rem.
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call i32 @srem_elide(i32 noundef 42)
  %call1 = call i32 @urem_elide(i32 noundef 42)
  %cmp = icmp ne i32 %call, %call1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
!14 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 3, type: !12)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocation(line: 4, column: 15, scope: !9)
!17 = !DILocalVariable(name: "rem", scope: !9, file: !1, line: 4, type: !12)
!23 = distinct !DISubprogram(name: "urem_elide", scope: !1, file: !1, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!24 = !DISubroutineType(types: !25)
!25 = !{!26, !26}
!26 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!27 = !DILocalVariable(name: "x", arg: 1, scope: !23, file: !1, line: 8, type: !26)
!28 = !DILocation(line: 0, scope: !23)
!29 = !DILocation(line: 9, column: 30, scope: !23)
!30 = !DILocalVariable(name: "remainder", scope: !23, file: !1, line: 9, type: !26)
