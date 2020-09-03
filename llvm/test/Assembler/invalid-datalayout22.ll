; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "v128:0:128"

; CHECK: ABI alignment specification must be >0 for non-aggregate types

