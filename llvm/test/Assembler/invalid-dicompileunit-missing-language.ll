; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:15: error: missing one of 'language' or 'sourceLanguageName', required for !DICompileUnit
!0 = distinct !DICompileUnit(file: !DIFile(filename: "a", directory: "b"))
