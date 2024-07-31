; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "p:64:64:64:128"
; CHECK: Index width cannot be larger than pointer width
