; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "a:"
; CHECK: Trailing separator in datalayout string
