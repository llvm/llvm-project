; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto2 run -o %t1.o %t.o -r %t.o,bar,pr
; RUN: llvm-readobj --symbols %t1.o.0 | FileCheck %s

; CHECK: Name: bar
; CHECK-NEXT: Value:
; CHECK-NEXT: Size:
; CHECK-NEXT: Binding: Weak
; CHECK-NEXT: Type: Function

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar() {
  ret void
}
