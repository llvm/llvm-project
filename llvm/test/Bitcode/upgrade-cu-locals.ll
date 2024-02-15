; Test moving of local imports from DICompileUnit's 'imports' to DISubprogram's 'retainedNodes'
;
; RUN: llvm-dis -o - %s.bc | FileCheck %s

%"struct.ns::t1" = type { i8 }

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare dso_local void @_Z3pinv() local_unnamed_addr

define dso_local i32 @main() local_unnamed_addr !dbg !23 {
entry:
  call void @llvm.dbg.declare(metadata ptr undef, metadata !39, metadata !DIExpression()), !dbg !40
  call void @_Z3pinv(), !dbg !42
  ret i32 0, !dbg !43
}

define dso_local i32 @main2() local_unnamed_addr !dbg !29 {
  ret i32 0
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0, !16}
!llvm.ident = !{!33, !33}
!llvm.module.flags = !{!34, !35, !36, !37, !38}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !2,
; CHECK: !2 = !{!3}
; CHECK: !3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !4,
; CHECK: !4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t4"

; CHECK: !5 = !{}
; CHECK: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !7, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !5, nameTableKind: GNU)

; CHECK: !14 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !18)
; CHECK: !18 = !{!19}
; CHECK: !19 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !23,
; CHECK: !20 = !DILexicalBlock(scope: !21, file: !7, line: 7, column: 35)
; CHECK: !21 = !DILexicalBlock(scope: !22, file: !7, line: 7, column: 35)
; CHECK: !22 = !DILexicalBlock(scope: !14, file: !7, line: 7, column: 35)
; CHECK: !23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t5", scope: !20,

; CHECK: !25 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 3, type: !26, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
; CHECK: !28 = !{!29, !32, !34}
; CHECK: !29 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !25, entity: !30,
; CHECK: !30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1",
; CHECK: !32 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !25, entity: !33,
; CHECK: !33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2",
; CHECK: !34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !25, entity: !35,
; CHECK: !35 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3",

; CHECK: !40 = distinct !DISubprogram(name: "main2", scope: !7, file: !7, line: 10, type: !15, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !41)
; CHECK: !41 = !{!42, !44}
; CHECK: !42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !40, entity: !43,
; CHECK: !43 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t6"
; CHECK: !44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !40, entity: !45,
; CHECK: !45 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t7",


!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, nameTableKind: GNU)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!2 = !{!3, !10, !12, !14}

; Move t1 to DISubprogram f1
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !8, file: !1, line: 3)
!4 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", scope: !9, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t1E")
!9 = !DINamespace(name: "ns", scope: null)

; Move t2 to DISubprogram f1
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !11, file: !1, line: 3)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", scope: !9, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t2E")

; Move t3 to DISubprogram f1
!12 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !13, file: !1, line: 3)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", scope: !9, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t3E")

; Leave t4 in CU
!14 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !15, file: !1, line: 3)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t4", scope: !0, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t4E")
!16 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !17, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !18, nameTableKind: GNU)
!17 = !DIFile(filename: "b.cpp", directory: "/")
!18 = !{!19, !28, !31}

; Move t5 to DISubprogram main
!19 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !27, file: !1, line: 3)
!20 = !DILexicalBlock(scope: !21, file: !17, line: 7, column: 35)
!21 = !DILexicalBlock(scope: !22, file: !17, line: 7, column: 35)
!22 = !DILexicalBlock(scope: !23, file: !17, line: 7, column: 35)
!23 = distinct !DISubprogram(name: "main", scope: !17, file: !17, line: 2, type: !24, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !16, retainedNodes: !7)
!24 = !DISubroutineType(types: !25)
!25 = !{!26}
!26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!27 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t5", scope: !20, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t5E")

; Move t6 to DISubprogram main2
!28 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !29, entity: !30, file: !17, line: 3)
!29 = distinct !DISubprogram(name: "main2", scope: !17, file: !17, line: 10, type: !24, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !16, retainedNodes: !7)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t6", scope: !29, file: !17, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t6E")

; Move t7 to DISubprogram main2
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !29, entity: !32, file: !17, line: 3)
!32 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t7", scope: !29, file: !17, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t7E")
!33 = !{!"clang version 14.0.0"}
!34 = !{i32 7, !"Dwarf Version", i32 4}
!35 = !{i32 2, !"Debug Info Version", i32 3}
!36 = !{i32 1, !"wchar_size", i32 4}
!37 = !{i32 7, !"uwtable", i32 1}
!38 = !{i32 7, !"frame-pointer", i32 2}
!39 = !DILocalVariable(name: "v1", scope: !4, file: !1, line: 3, type: !8)
!40 = !DILocation(line: 3, column: 37, scope: !4, inlinedAt: !41)
!41 = distinct !DILocation(line: 3, column: 3, scope: !23)
!42 = !DILocation(line: 3, column: 41, scope: !4, inlinedAt: !41)
!43 = !DILocation(line: 4, column: 1, scope: !23)
