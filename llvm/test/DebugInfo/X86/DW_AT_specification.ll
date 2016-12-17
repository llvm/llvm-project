; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name {{.*}} "_ZN3foo3barEv"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN3foo3barEv"


@_ZZN3foo3barEvE1x = constant i32 0, align 4, !dbg !20

define void @_ZN3foo3barEv()  !dbg !5 {
entry:
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 ()", isOptimized: false, emissionKind: FullDebug, file: !27, enums: !1, retainedTypes: !1, globals: !18, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !6, scope: null, type: !7, declaration: !11)
!6 = !DIFile(filename: "nsNativeAppSupportBase.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, flags: DIFlagFwdDecl, file: !27)
!11 = !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !6, scope: !12, type: !7)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "foo", line: 1, size: 8, align: 8, file: !27, elements: !13)
!13 = !{!11}
!18 = !{!20}
!20 = !DIGlobalVariable(name: "x", line: 5, isLocal: true, isDefinition: true, scope: !5, file: !6, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!22 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!25 = !DILocation(line: 6, column: 1, scope: !26)
!26 = distinct !DILexicalBlock(line: 4, column: 17, file: !6, scope: !5)
!27 = !DIFile(filename: "nsNativeAppSupportBase.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/toolkit/library")
!28 = !{i32 1, !"Debug Info Version", i32 3}
