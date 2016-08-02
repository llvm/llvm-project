; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; This was pulled from clang's debug-info-template-recursive.cpp test.
; class base { };

; template <class T> class foo : public base  {
;   void operator=(const foo r) { }
; };

; class bar : public foo<void> { };
; bar filters;

; CHECK: DW_TAG_template_type_parameter [{{.*}}]
; CHECK-NEXT: DW_AT_name{{.*}}"T"
; CHECK-NOT: DW_AT_type
; CHECK: NULL

%class.bar = type { i8 }

@filters = global %class.bar zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (trunk 187958) (llvm/trunk 187964)", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "debug-info-template-recursive.cpp", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "filters", line: 10, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: %class.bar* @filters)
!5 = !DIFile(filename: "debug-info-template-recursive.cpp", directory: "/usr/local/google/home/echristo/tmp")
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "bar", line: 9, size: 8, align: 8, file: !1, elements: !7)
!7 = !{!8, !31}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_class_type, name: "foo<void>", line: 5, size: 8, align: 8, file: !1, elements: !10, templateParams: !29)
!10 = !{!11, !19, !25}
!11 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !9, baseType: !12)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "base", line: 3, size: 8, align: 8, file: !1, elements: !13)
!13 = !{!14}
!14 = !DISubprogram(name: "base", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !12, type: !15)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !12)
!19 = !DISubprogram(name: "operator=", linkageName: "_ZN3fooIvEaSES0_", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !1, scope: !9, type: !20)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22, !23}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !9)
!23 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!25 = !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !9, type: !26)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !22}
!29 = !{!30}
!30 = !DITemplateTypeParameter(name: "T", type: null)
!31 = !DISubprogram(name: "bar", line: 9, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 9, file: !1, scope: !6, type: !32)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !6)
!36 = !{i32 2, !"Dwarf Version", i32 3}
!37 = !{i32 1, !"Debug Info Version", i32 3}
