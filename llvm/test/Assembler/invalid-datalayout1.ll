; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "^"
; CHECK: Unknown specifier in datalayout string
