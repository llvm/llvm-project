; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "i32:32:24"

; CHECK: Invalid preferred alignment, must be a power of 2

