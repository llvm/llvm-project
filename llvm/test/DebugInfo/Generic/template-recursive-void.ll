; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

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
; CHECK: {{DW_TAG|NULL}}

source_filename = "test/DebugInfo/Generic/template-recursive-void.ll"

%class.bar = type { i8 }

@filters = global %class.bar zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!29}
!llvm.module.flags = !{!32, !33}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "filters", scope: null, file: !2, line: 10, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "debug-info-template-recursive.cpp", directory: "/usr/local/google/home/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "bar", file: !2, line: 9, size: 8, align: 8, elements: !4)
!4 = !{!5, !25}
!5 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !3, baseType: !6)
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "foo<void>", file: !2, line: 5, size: 8, align: 8, elements: !7, templateParams: !23)
!7 = !{!8, !15, !20}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_class_type, name: "base", file: !2, line: 3, size: 8, align: 8, elements: !10)
!10 = !{!11}
!11 = !DISubprogram(name: "base", scope: !9, file: !2, line: 3, type: !12, isLocal: false, isDefinition: false, scopeLine: 3, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!15 = !DISubprogram(name: "operator=", linkageName: "_ZN3fooIvEaSES0_", scope: !6, file: !2, line: 6, type: !16, isLocal: false, isDefinition: false, scopeLine: 6, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !19}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!19 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!20 = !DISubprogram(name: "foo", scope: !6, file: !2, line: 5, type: !21, isLocal: false, isDefinition: false, scopeLine: 5, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !18}
!23 = !{!24}
!24 = !DITemplateTypeParameter(name: "T", type: null)
!25 = !DISubprogram(name: "bar", scope: !3, file: !2, line: 9, type: !26, isLocal: false, isDefinition: false, scopeLine: 9, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4 (trunk 187958) (llvm/trunk 187964)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !30, retainedTypes: !30, globals: !31, imports: !30)
!30 = !{}
!31 = !{!0}
!32 = !{i32 2, !"Dwarf Version", i32 3}
!33 = !{i32 1, !"Debug Info Version", i32 3}

