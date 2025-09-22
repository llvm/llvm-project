; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s | grep DW_OP_breg7
; Use DW_OP_breg7 in variable's location expression if the variable is in a stack slot.

%struct.SVal = type { ptr, i32 }

define i32 @_Z3fooi4SVal(i32 %i, ptr noalias %location) nounwind ssp !dbg !17 {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %location, i64 0, metadata !25, metadata !DIExpression()), !dbg !24
  %0 = icmp ne i32 %i, 0, !dbg !27                ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1, !dbg !27

bb:                                               ; preds = %entry
  %1 = getelementptr inbounds %struct.SVal, ptr %location, i32 0, i32 1, !dbg !29 ; <ptr> [#uses=1]
  %2 = load i32, ptr %1, align 8, !dbg !29            ; <i32> [#uses=1]
  %3 = add i32 %2, %i, !dbg !29                   ; <i32> [#uses=1]
  br label %bb2, !dbg !29

bb1:                                              ; preds = %entry
  %4 = getelementptr inbounds %struct.SVal, ptr %location, i32 0, i32 1, !dbg !30 ; <ptr> [#uses=1]
  %5 = load i32, ptr %4, align 8, !dbg !30            ; <i32> [#uses=1]
  %6 = sub i32 %5, 1, !dbg !30                    ; <i32> [#uses=1]
  br label %bb2, !dbg !30

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ %3, %bb ], [ %6, %bb1 ]         ; <i32> [#uses=1]
  br label %return, !dbg !29

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !29
}

define linkonce_odr void @_ZN4SValC1Ev(ptr %this) nounwind ssp align 2 !dbg !16 {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata ptr %this, i64 0, metadata !31, metadata !DIExpression()), !dbg !34
  store ptr null, ptr %this, align 8, !dbg !34
  %0 = getelementptr inbounds %struct.SVal, ptr %this, i32 0, i32 1, !dbg !34 ; <ptr> [#uses=1]
  store i32 0, ptr %0, align 8, !dbg !34
  br label %return, !dbg !34

return:                                           ; preds = %entry
  ret void, !dbg !35
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp !dbg !20 {
entry:
  %0 = alloca %struct.SVal                        ; <ptr> [#uses=3]
  %v = alloca %struct.SVal                        ; <ptr> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %v, metadata !38, metadata !DIExpression()), !dbg !41
  call void @_ZN4SValC1Ev(ptr %v) nounwind, !dbg !41
  %1 = getelementptr inbounds %struct.SVal, ptr %v, i32 0, i32 1, !dbg !42 ; <ptr> [#uses=1]
  store i32 1, ptr %1, align 8, !dbg !42
  %2 = load ptr, ptr %v, align 8, !dbg !43            ; <ptr> [#uses=1]
  store ptr %2, ptr %0, align 8, !dbg !43
  %3 = getelementptr inbounds %struct.SVal, ptr %0, i32 0, i32 1, !dbg !43 ; <ptr> [#uses=1]
  %4 = getelementptr inbounds %struct.SVal, ptr %v, i32 0, i32 1, !dbg !43 ; <ptr> [#uses=1]
  %5 = load i32, ptr %4, align 8, !dbg !43            ; <i32> [#uses=1]
  store i32 %5, ptr %3, align 8, !dbg !43
  %6 = call i32 @_Z3fooi4SVal(i32 2, ptr noalias %0) nounwind, !dbg !43 ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata i32 %6, i64 0, metadata !44, metadata !DIExpression()), !dbg !43
  br label %return, !dbg !45

return:                                           ; preds = %entry
  ret i32 0, !dbg !45
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!49}
!46 = !{!16, !17, !20}

!0 = !DISubprogram(name: "SVal", line: 11, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, scopeLine: 11, file: !47, scope: !1, type: !14)
!1 = !DICompositeType(tag: DW_TAG_structure_type, name: "SVal", line: 1, size: 128, align: 64, file: !47, scope: !2, elements: !4)
!2 = !DIFile(filename: "small.cc", directory: "/Users/manav/R8248330")
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !47, enums: !48, retainedTypes: !48, imports:  null)
!4 = !{!5, !7, !0, !9}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "Data", line: 7, size: 64, align: 64, file: !47, scope: !1, baseType: !6)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !47, scope: !2, baseType: null)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "Kind", line: 8, size: 32, align: 32, offset: 64, file: !47, scope: !1, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!9 = !DISubprogram(name: "~SVal", line: 12, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, scopeLine: 12, file: !47, scope: !1, type: !10)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !47, scope: !2, baseType: !1)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !12}
!16 = distinct !DISubprogram(name: "SVal", linkageName: "_ZN4SValC1Ev", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 11, file: !47, scope: !1, type: !14)
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi4SVal", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 16, file: !47, scope: !2, type: !18)
!18 = !DISubroutineType(types: !19)
!19 = !{!13, !13, !1}
!20 = distinct !DISubprogram(name: "main", linkageName: "main", line: 23, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 23, file: !47, scope: !2, type: !21)
!21 = !DISubroutineType(types: !22)
!22 = !{!13}
!23 = !DILocalVariable(name: "i", line: 16, arg: 1, scope: !17, file: !2, type: !13)
!24 = !DILocation(line: 16, scope: !17)
!25 = !DILocalVariable(name: "location", line: 16, arg: 2, scope: !17, file: !2, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_reference_type, name: "SVal", size: 64, align: 64, file: !47, scope: !2, baseType: !1)
!27 = !DILocation(line: 17, scope: !28)
!28 = distinct !DILexicalBlock(line: 16, column: 0, file: !47, scope: !17)
!29 = !DILocation(line: 18, scope: !28)
!30 = !DILocation(line: 20, scope: !28)
!31 = !DILocalVariable(name: "this", line: 11, arg: 1, scope: !16, file: !2, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_const_type, size: 64, align: 64, flags: DIFlagArtificial, file: !47, scope: !2, baseType: !33)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !47, scope: !2, baseType: !1)
!34 = !DILocation(line: 11, scope: !16)
!35 = !DILocation(line: 11, scope: !36)
!36 = distinct !DILexicalBlock(line: 11, column: 0, file: !47, scope: !37)
!37 = distinct !DILexicalBlock(line: 11, column: 0, file: !47, scope: !16)
!38 = !DILocalVariable(name: "v", line: 24, scope: !39, file: !2, type: !1)
!39 = distinct !DILexicalBlock(line: 23, column: 0, file: !47, scope: !40)
!40 = distinct !DILexicalBlock(line: 23, column: 0, file: !47, scope: !20)
!41 = !DILocation(line: 24, scope: !39)
!42 = !DILocation(line: 25, scope: !39)
!43 = !DILocation(line: 26, scope: !39)
!44 = !DILocalVariable(name: "k", line: 26, scope: !39, file: !2, type: !13)
!45 = !DILocation(line: 27, scope: !39)
!47 = !DIFile(filename: "small.cc", directory: "/Users/manav/R8248330")
!48 = !{}
!49 = !{i32 1, !"Debug Info Version", i32 3}
