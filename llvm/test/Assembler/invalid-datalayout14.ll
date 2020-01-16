; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "i64:64:16"
; CHECK: Preferred alignment cannot be less than the ABI alignment
