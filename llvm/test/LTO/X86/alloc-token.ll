; --- Full LTO ---
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto -exported-symbol=main -o %t.out %t.bc
; RUN: llvm-objdump -d -r %t.out | FileCheck %s
; --- ThinLTO ---
; RUN: opt -module-summary -o %t.thin.bc %s
; RUN: llvm-lto2 run %t.thin.bc -o %t.thin.out \
; RUN:   -r=%t.thin.bc,main,plx \
; RUN:   -r=%t.thin.bc,_Znwm, \
; RUN:   -r=%t.thin.bc,sink,pl
; RUN: llvm-objdump -d -r %t.thin.out.1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znwm(i64)

@sink = global ptr zeroinitializer

; CHECK-LABEL: <main>:
; CHECK: callq
; CHECK-NEXT: R_X86_64_PLT32 __alloc_token__Znwm
define void @main() sanitize_alloc_token {
  %call = call ptr @_Znwm(i64 8)
  store volatile ptr %call, ptr @sink
  ret void
}
