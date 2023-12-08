; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "p:64:24:64"

; CHECK: Pointer ABI alignment must be a power of 2

