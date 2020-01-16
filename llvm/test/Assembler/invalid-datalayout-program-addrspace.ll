; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Invalid address space, must be a 24-bit integer
target datalayout = "P16777216"
