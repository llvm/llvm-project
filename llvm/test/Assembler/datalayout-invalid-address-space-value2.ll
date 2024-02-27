; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: Trailing separator in datalayout string

target datalayout = "z:"