; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O2 -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Test back-to-back stores of TLS variables to ensure call sequences no
; longer overlap.

@__once_callable = external thread_local global ptr
@__once_call = external thread_local global ptr

define i64 @call_once(i64 %flag, ptr %ptr) {
entry:
  %var = alloca ptr, align 8
  store ptr %ptr, ptr %var, align 8
  store ptr %var, ptr @__once_callable, align 8
  store ptr @__once_call_impl, ptr @__once_call, align 8
  ret i64 %flag
}

; CHECK-LABEL: call_once:
; CHECK: addi 3, {{[0-9]+}}, __once_callable@got@tlsgd@l
; CHECK: bl __tls_get_addr(__once_callable@tlsgd)
; CHECK-NEXT: nop
; FIXME: We could check here for 'std {{[0-9]+}}, 0(3)', but that no longer
; works because, with new scheduling freedom, we create a copy of R3 based on the
; initial scheduling, but don't coalesce it again after we move the instructions
; so that the copy is no longer necessary.
; CHECK: addi 3, {{[0-9]+}}, __once_call@got@tlsgd@l
; CHECK: bl __tls_get_addr(__once_call@tlsgd)
; CHECK-NEXT: nop
; CHECK: std {{[0-9]+}}, 0(3)
; CHECK: mr 3, 30

declare void @__once_call_impl()
