; RUN: llvm-as < %s | llvm-dis | FileCheck %s
target datalayout = "p:52:64"
; CHECK: target datalayout = "p:52:64"