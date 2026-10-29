; RUN: opt -passes=correlated-propagation -S < %s | FileCheck %s

; When converting some signed instructions to unsigned, debug values need to be correctly
; updated to prevent incorrect values being reported
; See issues #218626 & #218627

define dso_local float @sitofp_to_uitofp(i32 noundef %x) #0 !dbg !11 {
; CHECK:      entry:
; CHECK-NEXT: %conv = uitofp nneg i32 %x to float
; CHECK-NEXT: #dbg_value(float poison, [[META1:![0-9]+]], !DIExpression(), [[META2:![0-9]+]])
entry:
  %conv = sitofp i32 %x to float
    #dbg_value(float %conv, !19, !DIExpression(), !17)
  %cmp = icmp sge i32 %x, 0
  %conv. = select i1 %cmp, float %conv, float 0.000000e+00
  ret float %conv.
}

define dso_local i32 @smin_to_umin(i32 noundef %x) #0 !dbg !23 {
; CHECK:      entry:
; CHECK-NEXT: %0 = call i32 @llvm.umin.i32(i32 %x, i32 42)
; CHECK-NEXT: #dbg_value(i32 poison, [[META4:![0-9]+]], !DIExpression(), [[META5:![0-9]+]])
entry:
  %elt.min = call i32 @llvm.smin.i32(i32 %x, i32 42)
    #dbg_value(i32 %elt.min, !29, !DIExpression(), !27)
  %cmp = icmp sge i32 %x, 0
  %elt.min. = select i1 %cmp, i32 %elt.min, i32 24
  ret i32 %elt.min.
}

declare i32 @llvm.smin.i32(i32, i32) #1

define dso_local i32 @main() #0 {
entry:
  %call = call float @sitofp_to_uitofp(i32 noundef -1)
  %call1 = call i32 @smin_to_umin(i32 noundef -1)
  %conv = sitofp i32 %call1 to float
  %cmp = fcmp une float %call, %conv
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

attributes #1 = { nocreateundeforpoison }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!2 = !{!3}
!3 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!11 = distinct !DISubprogram(name: "sitofp_to_uitofp", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!12 = !DISubroutineType(types: !13)
!13 = !{!3, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!17 = !DILocation(line: 0, scope: !11)
!19 = !DILocalVariable(name: "converted", scope: !11, file: !1, line: 2, type: !3)
!23 = distinct !DISubprogram(name: "smin_to_umin", scope: !1, file: !1, line: 6, type: !24, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!24 = !DISubroutineType(types: !25)
!25 = !{!14, !14}
!27 = !DILocation(line: 0, scope: !23)
!29 = !DILocalVariable(name: "minimum", scope: !23, file: !1, line: 7, type: !14)

; CHECK: [[META3:![0-9]+]] = distinct !DISubprogram(name: "sitofp_to_uitofp"
; CHECK: [[META1]] = !DILocalVariable(name: "converted", scope: [[META3]]
; CHECK: [[META2]] = !DILocation(line: 0, scope: [[META3]])
; CHECK: [[META6:![0-9]+]] = distinct !DISubprogram(name: "smin_to_umin"
; CHECK: [[META4]] = !DILocalVariable(name: "minimum", scope: [[META6]]
; CHECK: [[META5]] = !DILocation(line: 0, scope: [[META6]])
