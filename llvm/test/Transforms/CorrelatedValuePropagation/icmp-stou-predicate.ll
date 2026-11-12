; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When converting an icmp predicate from signed to unsigned,
; any debug values down-chain of that change need to be poisoned
; since the value they are based on may have changed.
; See issue #218620

; In this test, the slt becomes utl, changing the computed value
; of %cmp. This also changes the computed value of %conv, which
; relies on %cmp, and %conv is then used in a debug value. Since
; the value has changed, that debug value must now be poisoned

define dso_local i32 @icmp_signed_to_unsigned(i32 noundef %x) #0 !dbg !9 {
; CHECK:      entry:
; CHECK-NEXT: %cmp = icmp samesign ult i32 %x, 10
; CHECK-NEXT: %conv = zext i1 %cmp to i32
; CHECK-NEXT: #dbg_value(i32 poison, [[META1:![0-9]+]], !DIExpression(), [[META2:![0-9]+]])
entry:
  %cmp = icmp slt i32 %x, 10
  %conv = zext i1 %cmp to i32
    #dbg_value(i32 %conv, !17, !DIExpression(), !15)
  %cmp1 = icmp sge i32 %x, 0
  %conv. = select i1 %cmp1, i32 %conv, i32 0
  ret i32 %conv.
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "icmp_signed_to_unsigned", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "less", scope: !9, file: !1, line: 4, type: !12)
; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "icmp_signed_to_unsigned"
; CHECK: [[META1]] = !DILocalVariable(name: "less", scope: [[META3]]
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
