; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: not a number, or does not fit in an unsigned int

target datalayout = "z:neg-1"