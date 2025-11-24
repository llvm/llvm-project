; RUN: llvm-as %s -o %t.bc
;
; RUN: llvm-lto2 run -lto-alloc-token-mode=default %t.bc -o %t.out \
; RUN:   -r=%t.bc,main,plx \
; RUN:   -r=%t.bc,_Znwm, \
; RUN:   -r=%t.bc,sink,pl
; RUN: llvm-objdump -d -r %t.out.0 | FileCheck %s --check-prefixes=CHECK,DEFAULT
;
; RUN: llvm-lto2 run -lto-alloc-token-mode=default -alloc-token-fast-abi -alloc-token-max=1 %t.bc -o %t.out \
; RUN:   -r=%t.bc,main,plx \
; RUN:   -r=%t.bc,_Znwm, \
; RUN:   -r=%t.bc,sink,pl
; RUN: llvm-objdump -d -r %t.out.0 | FileCheck %s --check-prefixes=CHECK,FASTABI

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znwm(i64) #0

@sink = global ptr null

; CHECK-LABEL: <main>:
; CHECK: callq
; DEFAULT-NEXT: R_X86_64_PLT32 __alloc_token__Znwm
; FASTABI-NEXT: R_X86_64_PLT32 __alloc_token_0__Znwm
define void @main() sanitize_alloc_token {
  %call = call ptr @_Znwm(i64 8) #0, !alloc_token !0
  store volatile ptr %call, ptr @sink
  ret void
}

attributes #0 = { nobuiltin allocsize(0) }

!0 = !{!"int", i1 0}
