; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: {{ discriminator }}

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0, i32 %a1) local_unnamed_addr #0 !dbg !5 {
b0:
  call void @llvm.dbg.value(metadata i32 %a0, metadata !10, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.value(metadata i32 %a1, metadata !11, metadata !DIExpression()), !dbg !13
  %v0 = mul nsw i32 %a1, 30000, !dbg !14
  %v1 = icmp slt i32 %v0, %a0, !dbg !16
  %v2 = select i1 %v1, i32 %a0, i32 %v0, !dbg !16
  %v3 = add nsw i32 %v2, %a1, !dbg !17
  ret i32 %v3, !dbg !18
}

; Function Attrs: nounwind readnone
define i32 @f1() local_unnamed_addr #0 !dbg !19 {
b0:
  ret i32 0, !dbg !23
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "QuIC LLVM Hexagon Clang version hexagon-clang-82-1453 (based on LLVM 4.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/1.c", directory: "/local/mnt/workspace")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "long int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!11 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, line: 1, type: !8)
!12 = !DILocation(line: 1, column: 15, scope: !5)
!13 = !DILocation(line: 1, column: 23, scope: !5)
!14 = !DILocation(line: 2, column: 14, scope: !15)
!15 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 1)
!16 = !DILocation(line: 2, column: 1, scope: !5)
!17 = !DILocation(line: 4, column: 12, scope: !5)
!18 = !DILocation(line: 4, column: 3, scope: !5)
!19 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !20, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, unit: !0, variables: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{!22}
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!23 = !DILocation(line: 7, column: 14, scope: !19)
