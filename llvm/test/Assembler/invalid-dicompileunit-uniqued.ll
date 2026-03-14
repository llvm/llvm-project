; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:6: error: missing 'distinct', required for !DICompileUnit
!0 = !DICompileUnit(language: DW_LANG_C99, file: !DIFile(filename: "file", directory: "/dir"))
