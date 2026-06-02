; REQUIRES: x86-registered-target

; RUN: llvm-as < %s > %t1.bc
; RUN: llvm-lto2 run %t1.bc -r=%t1.bc,foo,plx -o %t.ref -select-save-temps=asm
; RUN: FileCheck %s < %t.asm.0.s

; CHECK: foo:

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  ret void
}
