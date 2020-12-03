; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Invalid address space, must be a 24-bit integer
target datalayout = "G16777216"
