; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:6: error: !DIArgList cannot appear outside of a function
!0 = !DIArgList()
