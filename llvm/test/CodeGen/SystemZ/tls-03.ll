; Test general-dynamic TLS accesses.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=CHECK-MAIN
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=CHECK-CP

@x = thread_local global i32 0

; Call __tls_get_offset to retrieve the symbol's TLS offset.
define ptr@foo() {
; CHECK-CP: .LCP{{.*}}:
; CHECK-CP: .quad x@TLSGD
;
; CHECK-MAIN-LABEL: foo:
; CHECK-MAIN-DAG: larl %r12, _GLOBAL_OFFSET_TABLE_
; CHECK-MAIN-DAG: lgrl %r2, .LCP{{.*}}
; CHECK-MAIN: brasl %r14, __tls_get_offset@PLT:tls_gdcall:x
; CHECK-MAIN: ear [[HIGH:%r[0-5]]], %a0
; CHECK-MAIN: sllg [[TP:%r[0-5]]], [[HIGH]], 32
; CHECK-MAIN: ear [[TP]], %a1
; CHECK-MAIN: agr %r2, [[TP]]
; CHECK-MAIN: br %r14
  ret ptr@x
}
