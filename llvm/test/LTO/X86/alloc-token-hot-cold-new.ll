; RUN: opt -module-summary -o %t.thin.bc %s
; RUN: llvm-lto2 run %t.thin.bc -o %t.thin.out \
; RUN:   -r=%t.thin.bc,main,plx \
; RUN:   -r=%t.thin.bc,_Znwm, \
; RUN:   -r=%t.thin.bc,sink,pl \
; RUN:   -supports-hot-cold-new -optimize-hot-cold-new
; RUN: llvm-objdump -d -r %t.thin.out.1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znwm(i64)

@sink = global ptr null

; CHECK-LABEL: <main>:
; CHECK: callq
; CHECK-NEXT: R_X86_64_PLT32 __alloc_token__Znwm12__hot_cold_t
define void @main() sanitize_alloc_token {
  %call = call ptr @_Znwm(i64 8) #0
  store volatile ptr %call, ptr @sink
  ret void
}

attributes #0 = { builtin allocsize(0) "memprof"="hot" }
