; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A16777216"
; CHECK: Invalid address space, must be a 24bit integer
