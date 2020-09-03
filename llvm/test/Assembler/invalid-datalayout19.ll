; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "p:0:32:32"

; CHECK: Invalid pointer size of 0 bytes

