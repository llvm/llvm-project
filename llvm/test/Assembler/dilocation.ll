; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !2, !3, !3, !4, !4, !5, !5, !6}
!named = !{!0, !2, !3, !4, !5, !6, !7, !8, !9}

!llvm.module.flags = !{!10}
!llvm.dbg.cu = !{!1}

; CHECK: !0 = distinct !DISubprogram(
!0 = distinct !DISubprogram(unit: !1)
; CHECK: !1 = distinct !DICompileUnit
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
; CHECK: !2 = !DIFile
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK-NEXT: !3 = !DILocation(line: 3, column: 7, scope: !0)
!3 = !DILocation(line: 3, column: 7, scope: !0)
!4 = !DILocation(scope: !0, column: 7, line: 3)

; CHECK-NEXT: !4 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !3)
!5 = !DILocation(scope: !0, inlinedAt: !3, column: 7, line: 3)
!6 = !DILocation(column: 7, line: 3, scope: !0, inlinedAt: !3)

; CHECK-NEXT: !5 = !DILocation(line: 0, scope: !0)
!7 = !DILocation(scope: !0)
!8 = !DILocation(scope: !0, column: 0, line: 0)

; CHECK-NEXT: !6 = !DILocation(line: 4294967295, column: 65535, scope: !0)
!9 = !DILocation(line: 4294967295, column: 65535, scope: !0)

!10 = !{i32 2, !"Debug Info Version", i32 3}
