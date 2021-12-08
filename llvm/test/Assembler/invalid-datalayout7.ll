; RUN: not --crash llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "p:48:52"
; CHECK: number of bits must be a byte width multiple
