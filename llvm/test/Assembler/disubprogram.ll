; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define void @_Z3foov() !dbg !9
define void @_Z3foov() !dbg !9 {
  ret void
}

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!5 = distinct !{}
!6 = distinct !{}

; CHECK: !7 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false)
!7 = distinct !DISubprogram()

; CHECK: !8 = !DISubprogram(scope: null, isLocal: false, isDefinition: false, isOptimized: false)
!8 = !DISubprogram(isDefinition: false)

; CHECK: !9 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, isLocal: true, isDefinition: true, scopeLine: 8, containingType: !4, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 10, flags: DIFlagPrototyped, isOptimized: true, templateParams: !5, declaration: !8, variables: !6)
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8, containingType: !4,
                            virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 10,
                            flags: DIFlagPrototyped, isOptimized: true,
                            templateParams: !5, declaration: !8, variables: !6)

!10 = !{i32 1, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!10}
