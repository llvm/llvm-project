; RUN: llc -mtriple=x86_64-linux < %s -filetype=obj | llvm-dwarfdump -debug-info - | FileCheck %s

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

source_filename = "test/DebugInfo/X86/concrete_out_of_line.ll"

define i32 @_ZN17nsAutoRefCnt7ReleaseEv() !dbg !19 {
entry:
  store i32 1, i32* null, align 4, !dbg !32
  tail call void @_Z8moz_freePv(i8* null) #0, !dbg !40
  ret i32 0
}

define void @_ZN17nsAutoRefCntD1Ev() !dbg !46 {
entry:
  tail call void @_Z8moz_freePv(i8* null) #0, !dbg !49
  ret void
}

declare void @_Z8moz_freePv(i8*)

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.1 ()", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "nsAutoRefCnt.ii", directory: "/Users/espindola/mozilla-central/obj-x86_64-apple-darwin11.2.0/netwerk/base/src")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "mRefCnt", scope: null, file: !1, line: 9, type: !6, isLocal: false, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "nsAutoRefCnt", file: !1, line: 2, size: 32, align: 32, elements: !7)
!7 = !{!8, !10, !15}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "mValue", scope: !6, file: !1, line: 7, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DISubprogram(name: "nsAutoRefCnt", scope: !6, file: !1, line: 3, type: !11, isLocal: false, isDefinition: false, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, align: 64, flags: DIFlagArtificial)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "nsAutoRefCnt", file: !1, line: 2, flags: DIFlagFwdDecl)
!15 = !DISubprogram(name: "operator=", linkageName: "_ZN12nsAutoRefCntaSEi", scope: !6, file: !1, line: 4, type: !16, isLocal: false, isDefinition: false, scopeLine: 4, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!9, !13, !9}
!18 = !{i32 1, !"Debug Info Version", i32 3}
!19 = distinct !DISubprogram(name: "Release", linkageName: "_ZN17nsAutoRefCnt7ReleaseEv", scope: null, file: !1, line: 14, type: !20, isLocal: false, isDefinition: true, scopeLine: 14, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !24, variables: !30)
!20 = !DISubroutineType(types: !21)
!21 = !{!9, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64, align: 64, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "nsAutoRefCnt", file: !1, line: 10, flags: DIFlagFwdDecl)
!24 = !DISubprogram(name: "Release", linkageName: "_ZN17nsAutoRefCnt7ReleaseEv", scope: !25, file: !1, line: 11, type: !20, isLocal: false, isDefinition: false, scopeLine: 11, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!25 = !DICompositeType(tag: DW_TAG_class_type, name: "nsAutoRefCnt", file: !1, line: 10, size: 8, align: 8, elements: !26)
!26 = !{!24, !27}
!27 = !DISubprogram(name: "~nsAutoRefCnt", scope: !25, file: !1, line: 12, type: !28, isLocal: false, isDefinition: false, scopeLine: 12, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !22}
!30 = !{!31}
!31 = !DILocalVariable(name: "this", arg: 1, scope: !19, file: !1, line: 14, type: !22, flags: DIFlagArtificial)
!32 = !DILocation(line: 5, column: 5, scope: !33, inlinedAt: !38)
!33 = distinct !DILexicalBlock(scope: !34, file: !1, line: 4, column: 29)
!34 = distinct !DISubprogram(name: "operator=", linkageName: "_ZN12nsAutoRefCntaSEi", scope: null, file: !1, line: 4, type: !16, isLocal: false, isDefinition: true, scopeLine: 4, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !15, variables: !35)
!35 = !{!36, !37}
!36 = !DILocalVariable(name: "this", arg: 1, scope: !34, file: !1, line: 4, type: !13, flags: DIFlagArtificial)
!37 = !DILocalVariable(name: "aValue", arg: 2, scope: !34, file: !1, line: 4, type: !9)
!38 = !DILocation(line: 15, scope: !39)
!39 = distinct !DILexicalBlock(scope: !19, file: !1, line: 14, column: 34)
!40 = !DILocation(line: 19, column: 3, scope: !41, inlinedAt: !45)
!41 = distinct !DILexicalBlock(scope: !42, file: !1, line: 18, column: 41)
!42 = distinct !DISubprogram(name: "~nsAutoRefCnt", linkageName: "_ZN17nsAutoRefCntD2Ev", scope: null, file: !1, line: 18, type: !28, isLocal: false, isDefinition: true, scopeLine: 18, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !27, variables: !43)
!43 = !{!44}
!44 = !DILocalVariable(name: "this", arg: 1, scope: !42, file: !1, line: 18, type: !22, flags: DIFlagArtificial)
!45 = !DILocation(line: 18, column: 41, scope: !46, inlinedAt: !38)
!46 = distinct !DISubprogram(name: "~nsAutoRefCnt", linkageName: "_ZN17nsAutoRefCntD1Ev", scope: null, file: !1, line: 18, type: !28, isLocal: false, isDefinition: true, scopeLine: 18, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !27, variables: !47)
!47 = !{!48}
!48 = !DILocalVariable(name: "this", arg: 1, scope: !46, file: !1, line: 18, type: !22, flags: DIFlagArtificial)
!49 = !DILocation(line: 19, column: 3, scope: !41, inlinedAt: !50)
!50 = !DILocation(line: 18, column: 41, scope: !46)

