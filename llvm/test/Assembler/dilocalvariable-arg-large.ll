; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1}
!named = !{!0, !1}

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram()

; CHECK: !1 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)
!1 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)

!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !3,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2,
                             subprograms: !{!0})
!3 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!4 = !{i32 2, !"Debug Info Version", i32 3}
