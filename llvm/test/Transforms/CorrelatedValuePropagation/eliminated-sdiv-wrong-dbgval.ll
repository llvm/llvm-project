; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @sdiv_elide(i32 noundef %x) #0 !dbg !9 {
; CHECK:      #dbg_value(i32 %x, [[META14:![0-9]+]], !DIExpression(), [[META15:![0-9]+]])
; CHECK-NEXT: #dbg_value(i32 %x, [[META16:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_div, DW_OP_stack_value), [[META15]])
; CHECK-NEXT: %0 = add i32 %x, 41
;
entry:
    #dbg_value(i32 %x, !14, !DIExpression(), !15)
  %div = sdiv i32 %x, 42, !dbg !16
    #dbg_value(i32 %div, !17, !DIExpression(), !15)
  %0 = add i32 %x, 41
  %spec.select = icmp ult i32 %0, 83
  %cond = select i1 %spec.select, i32 %div, i32 24
  ret i32 %cond
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call i32 @sdiv_elide(i32 noundef 42)
  %cmp = icmp ne i32 %call, 24
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

attributes #0 = { noinline nounwind uwtable }

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
!14 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 3, type: !12)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocation(line: 4, column: 20, scope: !9)
!17 = !DILocalVariable(name: "quotient", scope: !9, file: !1, line: 4, type: !12)
