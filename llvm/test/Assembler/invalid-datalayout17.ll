; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "i16777216:16:16"
; CHECK: Invalid bit width, must be a 24bit integer
