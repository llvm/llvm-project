; RUN: opt -passes=correlated-propagation -S < %s | FileCheck %s
; CHECK: %{{[a-zA-Z0-9_]*}} = udiv i8 %x.nonneg, %y, !dbg ![[DBGLOC:[0-9]+]]
; CHECK-NEXT: %{{[a-zA-Z0-9_]*}}.neg = sub i8 0, %rem1, !dbg ![[DBGLOC]]

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #0

define void @test8_neg_neg(i8 %x, i8 %y) !dbg !5 {
  %c0 = icmp sle i8 %x, 0, !dbg !13
  call void @llvm.dbg.value(metadata i1 %c0, metadata !9, metadata !DIExpression()), !dbg !13
  call void @llvm.assume(i1 %c0), !dbg !14
  %c1 = icmp sge i8 %y, 0, !dbg !15
  call void @llvm.dbg.value(metadata i1 %c1, metadata !11, metadata !DIExpression()), !dbg !15
  call void @llvm.assume(i1 %c1), !dbg !16
  %rem = sdiv i8 %x, %y, !dbg !17
  call void @llvm.dbg.value(metadata i8 %rem, metadata !12, metadata !DIExpression()), !dbg !17
  ret void, !dbg !18
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "reduced.ll", directory: "/")
!5 = distinct !DISubprogram(name: "test8_neg_neg", linkageName: "test8_neg_neg", scope: null, file: !2, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !12}
!9 = !DILocalVariable(name: "1", scope: !5, file: !2, line: 1, type: !10)
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !2, line: 3, type: !10)
!12 = !DILocalVariable(name: "3", scope: !5, file: !2, line: 5, type: !10)
!13 = !DILocation(line: 1, column: 1, scope: !5)
!14 = !DILocation(line: 2, column: 1, scope: !5)
!15 = !DILocation(line: 3, column: 1, scope: !5)
!16 = !DILocation(line: 4, column: 1, scope: !5)
!17 = !DILocation(line: 5, column: 1, scope: !5)
!18 = !DILocation(line: 6, column: 1, scope: !5)