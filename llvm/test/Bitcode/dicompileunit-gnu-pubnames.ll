; RUN: llvm-as -disable-verify -o - %s | llvm-dis | FileCheck %s

!named = !{!0}
; CHECK: !DICompileUnit({{.*}}, gnuPubnames: true)
!0 = distinct !DICompileUnit(language: 12, file: !1, gnuPubnames: true)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
