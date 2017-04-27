; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define void @_Z3foov() !dbg !9
define void @_Z3foov() !dbg !9 {
  ret void
}

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!5 = distinct !{}
!6 = distinct !{}

; CHECK: !7 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !8)
!7 = distinct !DISubprogram(unit: !8)

!8 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)

; CHECK: !9 = !DISubprogram(scope: null, isLocal: false, isDefinition: false, isOptimized: false)
!9 = !DISubprogram(isDefinition: false)

; CHECK: !10 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, isLocal: true, isDefinition: true, scopeLine: 8, containingType: !4, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 10, thisAdjustment: 3, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: true, unit: !8, templateParams: !5, declaration: !9, variables: !6)
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8,
                            containingType: !4,
                            virtuality: DW_VIRTUALITY_pure_virtual,
                            virtualIndex: 10, thisAdjustment: 3, flags: DIFlagPrototyped | DIFlagNoReturn,
                            isOptimized: true, unit: !8, templateParams: !5,
                            declaration: !9, variables: !6)

; CHECK: !11 = distinct !DISubprogram
; CHECK-SAME: virtualIndex: 0,
!11 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8,
                            containingType: !4,
                            virtuality: DW_VIRTUALITY_pure_virtual,
                            virtualIndex: 0,
                            flags: DIFlagPrototyped, isOptimized: true,
                            unit: !8, templateParams: !5, declaration: !9,
                            variables: !6)

; CHECK: !12 = distinct !DISubprogram
; CHECK-NOT: virtualIndex
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8,
                            containingType: !4,
                            virtuality: DW_VIRTUALITY_none,
                            flags: DIFlagPrototyped, isOptimized: true,
                            unit: !8,
                            templateParams: !5, declaration: !9, variables: !6)

!13 = !{!4}
; CHECK: !13 = !{!4}
; CHECK: !14 = distinct !DISubprogram(name: "foo", scope: !1, file: !2, line: 1, type: !3, isLocal: true, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !8, thrownTypes: !13)
!14 = distinct !DISubprogram(name: "foo", scope: !1,
                            file: !2, line: 1, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 2, isOptimized: false,
                            unit: !8, thrownTypes: !13)

!15 = !{i32 1, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!15}
!llvm.dbg.cu = !{!8}
