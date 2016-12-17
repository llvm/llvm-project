; RUN: llc -mtriple=x86_64-linux < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; test that we add DW_AT_inline even when we only have concrete out of line
; instances.

; first check that we have a TAG_subprogram at a given offset and it has
; AT_inline.

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name {{.*}} "_ZN12nsAutoRefCntaSEi"

; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}} "~nsAutoRefCnt"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_decl_line {{.*}}18
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D2
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:      DW_AT
; CHECK: DW_TAG
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_decl_line {{.*}}18
; CHECK-NEXT:     DW_AT_{{.*}}linkage_name {{.*}}D1
; CHECK-NEXT:     DW_AT_specification {{.*}} "~nsAutoRefCnt"
; CHECK-NEXT:     DW_AT_inline
; CHECK-NOT:     DW_AT
; CHECK: [[D1_THIS_ABS:.*]]: DW_TAG_formal_parameter

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_specification {{.*}} "_ZN17nsAutoRefCnt7ReleaseEv"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN12nsAutoRefCntaSEi"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK-NOT: NULL
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"

; and then that a TAG_subprogram refers to it with AT_abstract_origin.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD1Ev"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_abstract_origin {{.*}} {[[D1_THIS_ABS]]} "this"
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_ZN17nsAutoRefCntD2Ev"


define i32 @_ZN17nsAutoRefCnt7ReleaseEv() !dbg !5 {
entry:
  store i32 1, i32* null, align 4, !dbg !50
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !54
  ret i32 0
}

define void @_ZN17nsAutoRefCntD1Ev() !dbg !23 {
entry:
  tail call void @_Z8moz_freePv(i8* null) nounwind, !dbg !57
  ret void
}

declare void @_Z8moz_freePv(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!60}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.1 ()", isOptimized: true, emissionKind: FullDebug, file: !59, enums: !1, retainedTypes: !1, globals: !47, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "Release", linkageName: "_ZN17nsAutoRefCnt7ReleaseEv", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 14, file: !6, scope: null, type: !7 , declaration: !12, variables: !20)
!6 = !DIFile(filename: "nsAutoRefCnt.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "nsAutoRefCnt", line: 10, flags: DIFlagFwdDecl, file: !59)
!12 = !DISubprogram(name: "Release", linkageName: "_ZN17nsAutoRefCnt7ReleaseEv", line: 11, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 11, file: !6, scope: !13, type: !7, variables: !18)
!13 = !DICompositeType(tag: DW_TAG_class_type, name: "nsAutoRefCnt", line: 10, size: 8, align: 8, file: !59, elements: !14)
!14 = !{!12, !15}
!15 = !DISubprogram(name: "~nsAutoRefCnt", line: 12, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 12, file: !6, scope: !13, type: !16, variables: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !10}
!18 = !{}
!20 = !{!22}
!22 = !DILocalVariable(name: "this", line: 14, arg: 1, flags: DIFlagArtificial, scope: !5, file: !6, type: !10)
!23 = distinct !DISubprogram(name: "~nsAutoRefCnt", linkageName: "_ZN17nsAutoRefCntD1Ev", line: 18, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 18, file: !6, scope: null, type: !16, declaration: !15, variables: !24)
!24 = !{!26}
!26 = !DILocalVariable(name: "this", line: 18, arg: 1, flags: DIFlagArtificial, scope: !23, file: !6, type: !10)
!27 = distinct !DISubprogram(name: "~nsAutoRefCnt", linkageName: "_ZN17nsAutoRefCntD2Ev", line: 18, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 18, file: !6, scope: null, type: !16, declaration: !15, variables: !28)
!28 = !{!30}
!30 = !DILocalVariable(name: "this", line: 18, arg: 1, flags: DIFlagArtificial, scope: !27, file: !6, type: !10)
!31 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN12nsAutoRefCntaSEi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 4, file: !6, scope: null, type: !32, declaration: !36, variables: !43)
!32 = !DISubroutineType(types: !33)
!33 = !{!9, !34, !9}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !35)
!35 = !DICompositeType(tag: DW_TAG_structure_type, name: "nsAutoRefCnt", line: 2, flags: DIFlagFwdDecl, file: !59)
!36 = !DISubprogram(name: "operator=", linkageName: "_ZN12nsAutoRefCntaSEi", line: 4, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 4, file: !6, scope: !37, type: !32, variables: !18)
!37 = !DICompositeType(tag: DW_TAG_class_type, name: "nsAutoRefCnt", line: 2, size: 32, align: 32, file: !59, elements: !38)
!38 = !{!39, !40, !36}
!39 = !DIDerivedType(tag: DW_TAG_member, name: "mValue", line: 7, size: 32, align: 32, file: !59, scope: !37, baseType: !9)
!40 = !DISubprogram(name: "nsAutoRefCnt", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !6, scope: !37, type: !41, variables: !18)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !34}
!43 = !{!45, !46}
!45 = !DILocalVariable(name: "this", line: 4, arg: 1, flags: DIFlagArtificial, scope: !31, file: !6, type: !34)
!46 = !DILocalVariable(name: "aValue", line: 4, arg: 2, scope: !31, file: !6, type: !9)
!47 = !{!49}
!49 = !DIGlobalVariable(name: "mRefCnt", line: 9, isLocal: false, isDefinition: true, scope: null, file: !6, type: !37)
!50 = !DILocation(line: 5, column: 5, scope: !51, inlinedAt: !52)
!51 = distinct !DILexicalBlock(line: 4, column: 29, file: !6, scope: !31)
!52 = !DILocation(line: 15, scope: !53)
!53 = distinct !DILexicalBlock(line: 14, column: 34, file: !6, scope: !5)
!54 = !DILocation(line: 19, column: 3, scope: !55, inlinedAt: !56)
!55 = distinct !DILexicalBlock(line: 18, column: 41, file: !6, scope: !27)
!56 = !DILocation(line: 18, column: 41, scope: !23, inlinedAt: !52)
!57 = !DILocation(line: 19, column: 3, scope: !55, inlinedAt: !58)
!58 = !DILocation(line: 18, column: 41, scope: !23)
!59 = !DIFile(filename: "nsAutoRefCnt.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src")
!60 = !{i32 1, !"Debug Info Version", i32 3}
