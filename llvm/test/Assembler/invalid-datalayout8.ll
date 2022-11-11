; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "e-p"
; CHECK: Missing size specification for pointer in datalayout string
