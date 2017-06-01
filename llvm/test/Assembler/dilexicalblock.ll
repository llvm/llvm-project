; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4, !5, !6, !7, !7, !8}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}

!llvm.module.flags = !{!11}
!llvm.dbg.cu = !{!0}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !1,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = distinct !DISubprogram(name: "foo", scope: !1, unit: !0)

; CHECK: !3 = !DILexicalBlock(scope: !2, file: !1, line: 7, column: 35)
!3 = !DILexicalBlock(scope: !2, file: !1, line: 7, column: 35)

; CHECK: !4 = !DILexicalBlock(scope: !2)
!4 = !DILexicalBlock(scope: !2)
!5 = !DILexicalBlock(scope: !2, file: null, line: 0, column: 0)

; CHECK: !5 = !DILexicalBlockFile(scope: !3, file: !1, discriminator: 0)
; CHECK: !6 = !DILexicalBlockFile(scope: !3, file: !1, discriminator: 1)
!6 = !DILexicalBlockFile(scope: !3, file: !1, discriminator: 0)
!7 = !DILexicalBlockFile(scope: !3, file: !1, discriminator: 1)

; CHECK: !7 = !DILexicalBlockFile(scope: !3, discriminator: 7)
!8 = !DILexicalBlockFile(scope: !3, discriminator: 7)
!9 = !DILexicalBlockFile(scope: !3, file: null, discriminator: 7)
!10 = distinct !{}

!11 = !{i32 2, !"Debug Info Version", i32 3}
