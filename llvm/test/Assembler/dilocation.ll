; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !4, !5, !5, !6, !6, !7, !7, !8, !9, !10}
!named = !{!0, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11}

!llvm.module.flags = !{!12}
!llvm.dbg.cu = !{!1}

; CHECK: !0 = distinct !DISubprogram(scope: null, type: !1, spFlags: DISPFlagDefinition, unit: !3)
!0 = distinct !DISubprogram(unit: !1, type: !13)
; CHECK: !1 = !DISubroutineType(types: !2)
; CHECK: !2 = !{null}
; CHECK: !3 = distinct !DICompileUnit
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
; CHECK: !4 = !DIFile
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: !5 = !DILocation(line: 3, column: 7, scope: !0)
!3 = !DILocation(line: 3, column: 7, scope: !0)
!4 = !DILocation(scope: !0, column: 7, line: 3)

; CHECK: !6 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !5)
!5 = !DILocation(scope: !0, inlinedAt: !3, column: 7, line: 3)
!6 = !DILocation(column: 7, line: 3, scope: !0, inlinedAt: !3)

; CHECK: !7 = !DILocation(line: 0, scope: !0)
!7 = !DILocation(scope: !0)
!8 = !DILocation(scope: !0, column: 0, line: 0)

; CHECK: !8 = !DILocation(line: 4294967295, column: 65535, scope: !0)
!9 = !DILocation(line: 4294967295, column: 65535, scope: !0)

!10 = !DILocation(scope: !0, column: 0, line: 0, isImplicitCode: true)
!11 = !DILocation(scope: !0, column: 0, line: 1, isImplicitCode: false)
; CHECK: !9 = !DILocation(line: 0, scope: !0, isImplicitCode: true)
; CHECK: !10 = !DILocation(line: 1, scope: !0)

!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DISubroutineType(types: !14)
!14 = !{null}
