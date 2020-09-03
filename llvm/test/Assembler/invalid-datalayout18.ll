; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "p:32:32:16"
; CHECK: Preferred alignment cannot be less than the ABI alignment
