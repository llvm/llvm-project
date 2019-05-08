; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define void @_Z3foov() !dbg !9
define void @_Z3foov() !dbg !9 {
  ret void
}

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !17, !19, !20, !21}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!5 = distinct !{}
!6 = distinct !{}

; CHECK: !7 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !8)
!7 = distinct !DISubprogram(unit: !8)

!8 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)

; CHECK: !9 = !DISubprogram(scope: null, spFlags: 0)
!9 = !DISubprogram(isDefinition: false)

; CHECK: !10 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, scopeLine: 8, containingType: !4, virtualIndex: 10, thisAdjustment: 3, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagPureVirtual | DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8, templateParams: !5, declaration: !9, retainedNodes: !6)
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8,
                            containingType: !4,
                            virtuality: DW_VIRTUALITY_pure_virtual,
                            virtualIndex: 10, thisAdjustment: 3, flags: DIFlagPrototyped | DIFlagNoReturn,
                            isOptimized: true, unit: !8, templateParams: !5,
                            declaration: !9, retainedNodes: !6)

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
                            retainedNodes: !6)

; CHECK: !12 = distinct !DISubprogram
; CHECK-NOT: virtualIndex
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                            file: !2, line: 7, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 8,
                            containingType: !4,
                            virtuality: DW_VIRTUALITY_none,
                            flags: DIFlagPrototyped, isOptimized: true,
                            unit: !8,
                            templateParams: !5, declaration: !9, retainedNodes: !6)

!13 = !{!4}
; CHECK: !13 = !{!4}
; CHECK: !14 = distinct !DISubprogram(name: "foo", scope: !1, file: !2, line: 1, type: !3, scopeLine: 2, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !8, thrownTypes: !13)
!14 = distinct !DISubprogram(name: "foo", scope: !1,
                            file: !2, line: 1, type: !3, isLocal: true,
                            isDefinition: true, scopeLine: 2, isOptimized: false,
                            unit: !8, thrownTypes: !13)

; CHECK: !15 = distinct !DISubprogram({{.*}}, flags: DIFlagPrototyped, spFlags: DISPFlagPureVirtual | DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized,
!15 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1,
                             file: !2, line: 7, type: !3, scopeLine: 8,
							 containingType: !4, virtualIndex: 0,
							 flags: DIFlagPrototyped,
							 spFlags: DISPFlagPureVirtual | DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized,
							 unit: !8, templateParams: !5, declaration: !9,
							 retainedNodes: !6)

!16 = !{i32 1, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!16}
!llvm.dbg.cu = !{!8}

; CHECK: !DISubprogram({{.*}}subroutine1{{.*}}, spFlags: DISPFlagDefinition | DISPFlagPure,
; CHECK: !DISubprogram({{.*}}subroutine2{{.*}}, spFlags: DISPFlagDefinition | DISPFlagElemental,
; CHECK: !DISubprogram({{.*}}subroutine3{{.*}}, spFlags: DISPFlagDefinition | DISPFlagRecursive,
; CHECK: !DISubprogram({{.*}}subroutine4{{.*}}, spFlags: DISPFlagDefinition | DISPFlagPure | DISPFlagElemental | DISPFlagRecursive,

!17 = distinct !DISubprogram(name: "subroutine1", scope: !1, file: !2, line: 1, type: !18, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagPure, unit: !8, retainedNodes: !6)
!18 = !DISubroutineType(types: !0)
!19 = distinct !DISubprogram(name: "subroutine2", scope: !1, file: !2, line: 5, type: !18, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagElemental, unit: !8, retainedNodes: !6)
!20 = distinct !DISubprogram(name: "subroutine3", scope: !1, file: !2, line: 9, type: !18, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagRecursive, unit: !8, retainedNodes: !6)
!21 = distinct !DISubprogram(name: "subroutine4", scope: !1, file: !2, line: 13, type: !18, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagPure | DISPFlagElemental | DISPFlagRecursive, unit: !8, retainedNodes: !6)
