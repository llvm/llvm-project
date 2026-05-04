; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:62: error: expected string constant
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: 5,
                             file: !DIFile(filename: "a", directory: "b"))
