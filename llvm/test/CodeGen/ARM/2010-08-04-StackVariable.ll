; RUN: llc -mtriple=arm-apple-darwin -O0 -o - %s | FileCheck %s
; Use DW_OP_breg in variable's location expression if the variable is in a stack slot.

; CHECK: @ DW_OP_breg

%struct.SVal = type { i8*, i32 }

define i32 @_Z3fooi4SVal(i32 %i, %struct.SVal* noalias %location) #0 !dbg !4 {
entry:
  %"alloca point" = bitcast i32 0 to i32
  br label %realentry

realentry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata %struct.SVal* %location, metadata !23, metadata !DIExpression()), !dbg !22
  %tmp = icmp ne i32 %i, 0, !dbg !25
  br i1 %tmp, label %bb, label %bb1, !dbg !25

bb:                                               ; preds = %entry
  %tmp1 = getelementptr inbounds %struct.SVal, %struct.SVal* %location, i32 0, i32 1, !dbg !27
  %tmp2 = load i32, i32* %tmp1, align 8, !dbg !27
  %tmp3 = add i32 %tmp2, %i, !dbg !27
  br label %bb2, !dbg !27

bb1:                                              ; preds = %entry
  %tmp4 = getelementptr inbounds %struct.SVal, %struct.SVal* %location, i32 0, i32 1, !dbg !28
  %tmp5 = load i32, i32* %tmp4, align 8, !dbg !28
  %tmp6 = sub i32 %tmp5, 1, !dbg !28
  br label %bb2, !dbg !28

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ %tmp3, %bb ], [ %tmp6, %bb1 ]
  br label %return, !dbg !27

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "small.cc", directory: "/Users/manav/R8248330")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi4SVal", scope: !1, file: !1, line: 16, type: !5, virtualIndex: 6, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !8}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "SVal", file: !1, line: 1, size: 64, align: 64, elements: !9)
!9 = !{!10, !12, !14, !18}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "Data", scope: !8, file: !1, line: 7, baseType: !11, size: 64, align: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, file: !1, baseType: null, size: 64, align: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "Kind", scope: !8, file: !1, line: 8, baseType: !13, size: 32, align: 32, offset: 64)
!13 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!14 = !DISubprogram(name: "SVal", scope: !8, file: !1, line: 11, type: !15, virtualIndex: 6, spFlags: 0)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, file: !1, baseType: !8, size: 64, align: 64, flags: DIFlagArtificial)
!18 = !DISubprogram(name: "~SVal", scope: !8, file: !1, line: 12, type: !19, virtualIndex: 6, spFlags: 0)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !17, !7}
!21 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 16, type: !7)
!22 = !DILocation(line: 16, scope: !4)
!23 = !DILocalVariable(name: "location", arg: 2, scope: !4, file: !1, line: 16, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_reference_type, name: "SVal", scope: !1, file: !1, baseType: !8, size: 32, align: 32)
!25 = !DILocation(line: 17, scope: !26)
!26 = distinct !DILexicalBlock(scope: !4, file: !1, line: 16)
!27 = !DILocation(line: 18, scope: !26)
!28 = !DILocation(line: 20, scope: !26)
