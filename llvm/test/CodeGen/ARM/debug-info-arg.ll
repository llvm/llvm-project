; RUN: llc < %s | FileCheck %s
; Test to check argument y's debug info uses FI
; Radar 10048772
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.tag_s = type { i32, i32, i32 }

define void @foo(%struct.tag_s* nocapture %this, %struct.tag_s* %c, i64 %x, i64 %y, %struct.tag_s* nocapture %ptr1, %struct.tag_s* nocapture %ptr2) nounwind ssp "frame-pointer"="all" !dbg !1 {
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %this, metadata !5, metadata !DIExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %c, metadata !13, metadata !DIExpression()), !dbg !21
  tail call void @llvm.dbg.value(metadata i64 %x, metadata !14, metadata !DIExpression()), !dbg !22
  tail call void @llvm.dbg.value(metadata i64 %y, metadata !17, metadata !DIExpression()), !dbg !23
;CHECK:	@DEBUG_VALUE: foo:y <- [DW_OP_plus_uconst 8] [$r7+0]
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %ptr1, metadata !18, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %ptr2, metadata !19, metadata !DIExpression()), !dbg !25
  %1 = icmp eq %struct.tag_s* %c, null, !dbg !26
  br i1 %1, label %3, label %2, !dbg !26

; <label>:2                                       ; preds = %0
  tail call void @foobar(i64 %x, i64 %y) nounwind, !dbg !28
  br label %3, !dbg !28

; <label>:3                                       ; preds = %0, %2
  ret void, !dbg !29
}

declare void @foobar(i64, i64)

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)", isOptimized: true, emissionKind: FullDebug, file: !32, enums: !{}, retainedTypes: !{}, imports:  null)
!1 = distinct !DISubprogram(name: "foo", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 11, file: !2, scope: !2, type: !3, retainedNodes: !31)
!2 = !DIFile(filename: "one.c", directory: "/Volumes/Athwagate/R10048772")
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocalVariable(name: "this", line: 11, arg: 1, scope: !1, file: !2, type: !6)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !0, baseType: !7)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "tag_s", line: 5, size: 96, align: 32, file: !32, scope: !0, elements: !8)
!8 = !{!9, !11, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 6, size: 32, align: 32, file: !32, scope: !7, baseType: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 7, size: 32, align: 32, offset: 32, file: !32, scope: !7, baseType: !10)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "z", line: 8, size: 32, align: 32, offset: 64, file: !32, scope: !7, baseType: !10)
!13 = !DILocalVariable(name: "c", line: 11, arg: 2, scope: !1, file: !2, type: !6)
!14 = !DILocalVariable(name: "x", line: 11, arg: 3, scope: !1, file: !2, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "UInt64", line: 1, file: !32, scope: !0, baseType: !16)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 32, encoding: DW_ATE_unsigned)
!17 = !DILocalVariable(name: "y", line: 11, arg: 4, scope: !1, file: !2, type: !15)
!18 = !DILocalVariable(name: "ptr1", line: 11, arg: 5, scope: !1, file: !2, type: !6)
!19 = !DILocalVariable(name: "ptr2", line: 11, arg: 6, scope: !1, file: !2, type: !6)
!20 = !DILocation(line: 11, column: 24, scope: !1)
!21 = !DILocation(line: 11, column: 44, scope: !1)
!22 = !DILocation(line: 11, column: 54, scope: !1)
!23 = !DILocation(line: 11, column: 64, scope: !1)
!24 = !DILocation(line: 11, column: 81, scope: !1)
!25 = !DILocation(line: 11, column: 101, scope: !1)
!26 = !DILocation(line: 12, column: 3, scope: !27)
!27 = distinct !DILexicalBlock(line: 11, column: 107, file: !2, scope: !1)
!28 = !DILocation(line: 13, column: 5, scope: !27)
!29 = !DILocation(line: 14, column: 1, scope: !27)
!31 = !{!5, !13, !14, !17, !18, !19}
!32 = !DIFile(filename: "one.c", directory: "/Volumes/Athwagate/R10048772")
!33 = !{i32 1, !"Debug Info Version", i32 3}
