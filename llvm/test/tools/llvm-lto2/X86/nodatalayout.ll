; RUN: llvm-as < %s > %t1.bc

; Reject input modules without a datalayout.
; RUN: not llvm-lto2 run %t1.bc -o %t.o \
; RUN:  -r %t1.bc,patatino,px 2>&1 | FileCheck %s

; CHECK: input module has no datalayout

target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  ret void
}
