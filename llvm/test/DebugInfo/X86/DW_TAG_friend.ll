; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Check that the friend tag is there and is followed by a DW_AT_friend that has a reference back.

; CHECK: [[BACK:0x[0-9a-f]*]]:   DW_TAG_class_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]       ( .debug_str[{{.*}}] = "A")
; CHECK: DW_TAG_friend
; CHECK-NEXT: DW_AT_friend [DW_FORM_ref4]   (cu + 0x{{[0-9a-f]*}} => {[[BACK]]})


%class.A = type { i32 }
%class.B = type { i32 }

@a = global %class.A zeroinitializer, align 4
@b = global %class.B zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.1 (trunk 153413) (llvm/trunk 153428)", isOptimized: false, emissionKind: FullDebug, file: !28, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5, !17}
!5 = !DIGlobalVariable(name: "a", line: 10, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: %class.A* @a)
!6 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 1, size: 32, align: 32, file: !28, elements: !8)
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, flags: DIFlagPrivate, file: !28, scope: !7, baseType: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DISubprogram(name: "A", line: 1, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !6, scope: !7, type: !12)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !7)
!17 = !DIGlobalVariable(name: "b", line: 11, isLocal: false, isDefinition: true, scope: null, file: !6, type: !18, variable: %class.B* @b)
!18 = !DICompositeType(tag: DW_TAG_class_type, name: "B", line: 5, size: 32, align: 32, file: !28, elements: !19)
!19 = !{!20, !21, !27}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 7, size: 32, align: 32, flags: DIFlagPrivate, file: !28, scope: !18, baseType: !10)
!21 = !DISubprogram(name: "B", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !6, scope: !18, type: !22)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !18)
!27 = !DIDerivedType(tag: DW_TAG_friend, file: !28, baseType: !7)
!28 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!29 = !{i32 1, !"Debug Info Version", i32 3}
