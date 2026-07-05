; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: target datalayout = "P1"
target datalayout = "P1"

