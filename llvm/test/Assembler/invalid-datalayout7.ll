; RUN: not llvm-as < %s 2>&1 | FileCheck %s
target datalayout = "p:52"
; CHECK: number of bits must be a byte width multiple
