; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:74: error: missing required field 'language'
!0 = distinct !DICompileUnit(file: !DIFile(filename: "a", directory: "b"))
