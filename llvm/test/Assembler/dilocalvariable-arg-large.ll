; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !3}
!named = !{!0, !3}

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!1}

!0 = distinct !DISubprogram(unit: !1)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: !3 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)
!3 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)

!4 = !{i32 2, !"Debug Info Version", i32 3}
