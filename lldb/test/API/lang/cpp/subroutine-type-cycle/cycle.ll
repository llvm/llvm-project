; Base is described by two DIEs, the way the parallel DWARF linker splits a
; class with a member that depends on a local type:
;
;   struct Base {
;     int x;
;     void erase(int *B) { erase(B, [](int &) {}); }
;     template <class F> void erase(int *B, F &&OnMoved);
;   };
;
; Resolving the type of erase<lambda> resolves the lambda, whose context is
; erase(int *). Its class is the other DIE, so the class is completed, which
; resolves the declaration of erase<lambda> and its lambda parameter again.

define void @erase() !dbg !20 {
  ret void, !dbg !25
}

define void @erase_lambda() !dbg !40 {
  ret void, !dbg !43
}

define i32 @main() {
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)

; The first DIE for Base holds x and erase(int *).
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, line: 1, size: 32, elements: !11)
!11 = !{!12, !13}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !1, line: 2, baseType: !4, size: 32)
!13 = !DISubprogram(name: "erase", linkageName: "erase", scope: !10, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !5}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)

; erase(int *) and its lambda.
!20 = distinct !DISubprogram(name: "erase", linkageName: "erase", scope: !10, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !13)
!21 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !20, file: !1, line: 3, size: 8, elements: !22)
!22 = !{}
!23 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !21, size: 64)
!25 = !DILocation(line: 3, scope: !20)

; The second DIE for Base holds only erase<lambda>(int *, lambda &&).
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, line: 1, size: 32, elements: !31)
!31 = !{!32}
!32 = !DISubprogram(name: "erase<lambda>", linkageName: "erase_lambda", scope: !30, file: !1, line: 4, type: !33, scopeLine: 4, flags: DIFlagPrototyped, spFlags: 0, templateParams: !36)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35, !5, !23}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !30, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !{!37}
!37 = !DITemplateTypeParameter(name: "F", type: !21)

!40 = distinct !DISubprogram(name: "erase<lambda>", linkageName: "erase_lambda", scope: !30, file: !1, line: 4, type: !33, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, templateParams: !36, declaration: !32, retainedNodes: !41)
!41 = !{!42}
!42 = !DILocalVariable(name: "OnMoved", arg: 3, scope: !40, file: !1, line: 4, type: !23)
!43 = !DILocation(line: 4, scope: !40)
