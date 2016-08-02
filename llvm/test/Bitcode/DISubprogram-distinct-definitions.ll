; RUN: llvm-dis < %s.bc | FileCheck %s
; Check that subprogram definitions are correctly upgraded to 'distinct'.
; Bitcode compiled from r245235 of the 3.7 release branch.

!named = !{!0}
!0 = distinct !DICompileUnit(language: 12, file: !1, subprograms: !2)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = !{!3}

; CHECK: = distinct !DISubprogram({{.*}}, isDefinition: true
!3 = !DISubprogram(name: "foo", isDefinition: true)
