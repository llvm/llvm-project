; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4, !5, !6, !7, !7}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!llvm.module.flags = !{!11}
!llvm.dbg.cu = !{!10}

!0 = distinct !{}
!1 = distinct !DISubprogram(name: "foo", scope: !2)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: !3 = !DILexicalBlock(scope: !1, file: !2, line: 7, column: 35)
!3 = !DILexicalBlock(scope: !1, file: !2, line: 7, column: 35)

; CHECK: !4 = !DILexicalBlock(scope: !1)
!4 = !DILexicalBlock(scope: !1)
!5 = !DILexicalBlock(scope: !1, file: null, line: 0, column: 0)

; CHECK: !5 = !DILexicalBlockFile(scope: !3, file: !2, discriminator: 0)
; CHECK: !6 = !DILexicalBlockFile(scope: !3, file: !2, discriminator: 1)
!6 = !DILexicalBlockFile(scope: !3, file: !2, discriminator: 0)
!7 = !DILexicalBlockFile(scope: !3, file: !2, discriminator: 1)

; CHECK: !7 = !DILexicalBlockFile(scope: !3, discriminator: 7)
!8 = !DILexicalBlockFile(scope: !3, discriminator: 7)
!9 = !DILexicalBlockFile(scope: !3, file: null, discriminator: 7)

!10 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2,
                             subprograms: !{!1})
!11 = !{i32 2, !"Debug Info Version", i32 3}
