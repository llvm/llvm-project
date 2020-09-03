; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "m."
; CHECK: Unexpected trailing characters after mangling specifier in datalayout string
