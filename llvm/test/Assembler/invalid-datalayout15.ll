; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "i64:16:16777216"
; CHECK: Invalid preferred alignment, must be a 16bit integer
