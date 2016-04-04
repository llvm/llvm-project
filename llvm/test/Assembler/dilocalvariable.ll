; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!llvm.module.flags = !{!10}
!llvm.dbg.cu = !{!9}

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

!9 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2,
                             subprograms: !{!0})
!10 = !{i32 2, !"Debug Info Version", i32 3}
