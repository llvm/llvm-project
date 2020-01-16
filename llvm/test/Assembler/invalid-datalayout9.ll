; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "e-p:64"
; CHECK: Missing alignment specification for pointer in datalayout string
