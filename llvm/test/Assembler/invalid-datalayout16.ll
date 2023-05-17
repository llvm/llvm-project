; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "i64:16777216:16777216"
; CHECK: Invalid ABI alignment, must be a 16bit integer
