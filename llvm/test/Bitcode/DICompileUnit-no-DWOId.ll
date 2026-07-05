; The input uses the older (r236428) form without a dwoId field.  This should
; default to 0, which is not displayed at all in the textual representation.
;
; RUN: llvm-dis %s.bc -o - | FileCheck %s
; CHECK: distinct !DICompileUnit
; CHECK-NOT: dwoId:
!named = !{!0}
!0 = !DICompileUnit(language: 12, file: !1)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
