; RUN: llc -march=ppc64 -mcpu=pwr7 -O2 -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Test back-to-back stores of TLS variables to ensure call sequences no
; longer overlap.

@__once_callable = external thread_local global i8**
@__once_call = external thread_local global void ()*

define i64 @call_once(i64 %flag, i8* %ptr) {
entry:
  %var = alloca i8*, align 8
  store i8* %ptr, i8** %var, align 8
  store i8** %var, i8*** @__once_callable, align 8
  store void ()* @__once_call_impl, void ()** @__once_call, align 8
  ret i64 %flag
}

; CHECK-LABEL: call_once:
; CHECK: addis 3, 2, __once_callable@got@tlsgd@ha
; CHECK: addi 3, 3, __once_callable@got@tlsgd@l
; CHECK: bl __tls_get_addr(__once_callable@tlsgd)
; CHECK-NEXT: nop
; CHECK: std {{[0-9]+}}, 0(3)
; CHECK: addis 3, 2, __once_call@got@tlsgd@ha
; CHECK: addi 3, 3, __once_call@got@tlsgd@l
; CHECK: bl __tls_get_addr(__once_call@tlsgd)
; CHECK-NEXT: nop
; CHECK: std {{[0-9]+}}, 0(3)

declare void @__once_call_impl()
