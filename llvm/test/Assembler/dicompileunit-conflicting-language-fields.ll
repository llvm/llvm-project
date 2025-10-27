; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:15: error: can only specify one of 'language' and 'sourceLanguageName' on !DICompileUnit
!0 = distinct !DICompileUnit(language: DW_LANG_C, sourceLanguageName: DW_LNAME_C, file: !DIFile(filename: "a", directory: "b"))
