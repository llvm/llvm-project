; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "m:v"
; CHECK: Unknown mangling in datalayout string
