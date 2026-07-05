; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:36: error: 'file' cannot be null
!0 = distinct !DICompileUnit(file: null)
