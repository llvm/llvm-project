; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "p16777216:64:64:64"
; CHECK: Invalid address space, must be a 24-bit integer
