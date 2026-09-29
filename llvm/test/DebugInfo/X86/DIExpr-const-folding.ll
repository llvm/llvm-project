; RUN: llc -mtriple=x86_64 -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -mtriple=x86_64 -filetype=obj -fast-isel < %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s

;; The important thing is the DW_OP_lit2 with no converts.
;; TODO: Make this work with global isel
;; Indirectly related FIXME: Should be able to emit DW_AT_const_value instead.

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location (DW_OP_lit2, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("bIsShowingCollision")

%class.UClient = type { ptr, %struct.FFlags }
%class.UWorld = type { i16 }
%struct.FFlags = type { [9 x i8], i32 }

define dso_local void @_ZN7UClient13ToggleVolumesEv(ptr nocapture nonnull align 8 dereferenceable(24) %this) local_unnamed_addr align 2 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i72 2, metadata !43, metadata !DIExpression(DW_OP_LLVM_convert, 72, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !48
  %0 = load ptr, ptr %this, align 8, !dbg !49, !tbaa !51
  store i16 2, ptr %0, align 1, !dbg !59
  ret void, !dbg !60
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "min-test-3.cpp", directory: "/home/probinson/projects/scratch/tc8251")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang version 13.0.0"}
!8 = distinct !DISubprogram(name: "ToggleVolumes", linkageName: "_ZN7UClient13ToggleVolumesEv", scope: !9, file: !1, line: 39, type: !37, scopeLine: 40, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !36, retainedNodes: !40)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "UClient", file: !1, line: 31, size: 192, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !10, identifier: "_ZTS7UClient")
!10 = !{!11, !20, !36}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "World", scope: !9, file: !1, line: 34, baseType: !12, size: 64, flags: DIFlagPublic)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "UWorld", file: !1, line: 8, size: 16, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS6UWorld")
!14 = !{!15, !18, !19}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "bCollision", scope: !13, file: !1, line: 11, baseType: !16, size: 1, flags: DIFlagPublic | DIFlagBitField, extraData: i64 0)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8", file: !1, line: 1, baseType: !17)
!17 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "dummyA", scope: !13, file: !1, line: 12, baseType: !16, size: 7, offset: 1, flags: DIFlagPublic | DIFlagBitField, extraData: i64 0)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "dummyB", scope: !13, file: !1, line: 13, baseType: !16, size: 1, offset: 8, flags: DIFlagPublic | DIFlagBitField, extraData: i64 0)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "EngineShowFlags", scope: !9, file: !1, line: 35, baseType: !21, size: 128, offset: 64, flags: DIFlagPublic)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FFlags", file: !1, line: 16, size: 128, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !22, identifier: "_ZTS6FFlags")
!22 = !{!23, !26, !27, !28, !29, !30, !32}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "Volumes", scope: !21, file: !1, line: 18, baseType: !24, size: 1, flags: DIFlagBitField, extraData: i64 0)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32", file: !1, line: 2, baseType: !25)
!25 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "Collision", scope: !21, file: !1, line: 19, baseType: !24, size: 1, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "dummy1", scope: !21, file: !1, line: 20, baseType: !24, size: 30, offset: 2, flags: DIFlagBitField, extraData: i64 0)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "dummy2", scope: !21, file: !1, line: 21, baseType: !24, size: 32, offset: 32, flags: DIFlagBitField, extraData: i64 0)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "dummy3", scope: !21, file: !1, line: 22, baseType: !24, size: 1, offset: 64, flags: DIFlagBitField, extraData: i64 0)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "CustomShowFlags", scope: !21, file: !1, line: 24, baseType: !31, size: 32, offset: 96)
!31 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!32 = !DISubprogram(name: "FFlags", scope: !21, file: !1, line: 25, type: !33, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DISubprogram(name: "ToggleVolumes", linkageName: "_ZN7UClient13ToggleVolumesEv", scope: !9, file: !1, line: 36, type: !37, scopeLine: 36, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !39}
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!40 = !{!41, !43}
!41 = !DILocalVariable(name: "this", arg: 1, scope: !8, type: !42, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!43 = !DILocalVariable(name: "bIsShowingCollision", scope: !44, file: !1, line: 45, type: !46)
!44 = distinct !DILexicalBlock(scope: !45, file: !1, line: 42, column: 2)
!45 = distinct !DILexicalBlock(scope: !8, file: !1, line: 41, column: 6)
!46 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !47)
!47 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!48 = !DILocation(line: 0, scope: !44)
!49 = !DILocation(line: 51, column: 8, scope: !50)
!50 = distinct !DILexicalBlock(scope: !44, file: !1, line: 51, column: 8)
!51 = !{!52, !53, i64 0}
!52 = !{!"_ZTS7UClient", !53, i64 0, !56, i64 8}
!53 = !{!"any pointer", !54, i64 0}
!54 = !{!"omnipotent char", !55, i64 0}
!55 = !{!"Simple C++ TBAA"}
!56 = !{!"_ZTS6FFlags", !57, i64 0, !57, i64 0, !57, i64 0, !57, i64 4, !57, i64 8, !57, i64 12}
!57 = !{!"int", !54, i64 0}
!58 = !DILocation(line: 52, column: 12, scope: !50)
!59 = !DILocation(line: 52, column: 23, scope: !50)
!60 = !DILocation(line: 55, column: 1, scope: !8)
