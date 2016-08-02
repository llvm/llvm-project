; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DISubprogram()
!1 = distinct !{}
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DILocation(scope: !0)

; CHECK: !5 = !DILocalVariable(name: "foo", arg: 3, scope: !0, file: !2, line: 7, type: !3, flags: DIFlagArtificial)
; CHECK: !6 = !DILocalVariable(name: "foo", scope: !0, file: !2, line: 7, type: !3, flags: DIFlagArtificial)
!5 = !DILocalVariable(name: "foo", arg: 3,
                      scope: !0, file: !2, line: 7, type: !3,
                      flags: DIFlagArtificial)
!6 = !DILocalVariable(name: "foo", scope: !0,
                      file: !2, line: 7, type: !3, flags: DIFlagArtificial)

; CHECK: !7 = !DILocalVariable(arg: 1, scope: !0)
; CHECK: !8 = !DILocalVariable(scope: !0)
!7 = !DILocalVariable(scope: !0, arg: 1)
!8 = !DILocalVariable(scope: !0)
