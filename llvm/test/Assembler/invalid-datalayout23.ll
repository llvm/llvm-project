; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "i32:24:32"

; CHECK: Invalid ABI alignment, must be a power of 2

