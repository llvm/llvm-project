; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: Invalid ABI alignment, i8 must be naturally aligned

target datalayout = "i8:16"
