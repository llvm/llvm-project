; RUN: opt -passes=licm < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<target-ir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

; RUN: opt -passes=licm < %s -S --try-experimental-debuginfo-iterators | FileCheck %s

define void @dgefa() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond.backedge, %entry
  br i1 undef, label %if.then, label %for.cond.backedge, !dbg !11

for.cond.backedge:                                ; preds = %for.body61, %for.body61.us, %for.body
  br i1 undef, label %for.end104, label %for.body, !dbg !15

if.then:                                          ; preds = %for.body
  br i1 undef, label %if.then27, label %if.end.if.end.split_crit_edge.critedge, !dbg !16

if.then27:                                        ; preds = %if.then
; CHECK: tail call void @llvm.dbg.value
  tail call void @llvm.dbg.value(metadata double undef, metadata !19, metadata !DIExpression()), !dbg !21
  br label %for.body61.us

if.end.if.end.split_crit_edge.critedge:           ; preds = %if.then
  br label %for.body61

for.body61.us:                                    ; preds = %for.body61.us, %if.then27
  br i1 undef, label %for.cond.backedge, label %for.body61.us, !dbg !23

for.body61:                                       ; preds = %for.body61, %if.end.if.end.split_crit_edge.critedge
  br i1 undef, label %for.cond.backedge, label %for.body61, !dbg !23

for.end104:                                       ; preds = %for.cond.backedge
  ret void, !dbg !24
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!26}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "idamax", line: 112, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !25, scope: !1, type: !3)
!1 = !DIFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/Benchmarks/CoyoteBench/lpbench.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 127169)", isOptimized: true, emissionKind: FullDebug, file: !25)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "dscal", line: 206, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !25, scope: !1, type: !7)
!7 = !DISubroutineType(types: !{null})
!9 = distinct !DISubprogram(name: "daxpy", line: 230, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !25, scope: !1, type: !7)
!10 = distinct !DISubprogram(name: "dgefa", line: 267, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !25, scope: !1, type: !7)
!11 = !DILocation(line: 281, column: 9, scope: !12)
!12 = distinct !DILexicalBlock(line: 272, column: 5, file: !25, scope: !13)
!13 = distinct !DILexicalBlock(line: 271, column: 5, file: !25, scope: !14)
!14 = distinct !DILexicalBlock(line: 267, column: 1, file: !25, scope: !10)
!15 = !DILocation(line: 271, column: 5, scope: !14)
!16 = !DILocation(line: 284, column: 10, scope: !17)
!17 = distinct !DILexicalBlock(line: 282, column: 9, file: !25, scope: !12)
!18 = !{double undef}
!19 = !DILocalVariable(name: "temp", line: 268, scope: !14, file: !1, type: !20)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!21 = !DILocation(line: 286, column: 14, scope: !22)
!22 = distinct !DILexicalBlock(line: 285, column: 13, file: !25, scope: !17)
!23 = !DILocation(line: 296, column: 13, scope: !17)
!24 = !DILocation(line: 313, column: 1, scope: !14)
!25 = !DIFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/Benchmarks/CoyoteBench/lpbench.c", directory: "/private/tmp")
!26 = !{i32 1, !"Debug Info Version", i32 3}
