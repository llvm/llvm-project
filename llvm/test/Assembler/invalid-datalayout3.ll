; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "n0"
; CHECK: Zero width native integer type in datalayout string
