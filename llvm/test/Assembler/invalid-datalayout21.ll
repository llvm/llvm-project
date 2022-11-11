; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "p:64:64:24"

; CHECK: Pointer preferred alignment must be a power of 2

