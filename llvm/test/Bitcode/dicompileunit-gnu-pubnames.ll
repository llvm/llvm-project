; RUN: llvm-as -disable-verify -o - %s | llvm-dis | FileCheck %s

!named = !{!0}
; CHECK: !DICompileUnit({{.*}}, nameTableKind: GNU)
!0 = distinct !DICompileUnit(language: 12, file: !1, nameTableKind: GNU)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
