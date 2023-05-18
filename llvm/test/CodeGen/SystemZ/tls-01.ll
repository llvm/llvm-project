; Test local-exec TLS accesses.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-MAIN
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-CP

@x = dso_local thread_local global i32 0

; The offset must be loaded from the constant pool.  It doesn't really
; matter whether we use LARL/AG or LGRL/AGR for the last part.
define dso_local ptr@foo() {
; CHECK-CP: .LCP{{.*}}:
; CHECK-CP: .quad x@NTPOFF
;
; CHECK-MAIN-LABEL: foo:
; CHECK-MAIN: ear [[HIGH:%r[0-5]]], %a0
; CHECK-MAIN: sllg %r2, [[HIGH]], 32
; CHECK-MAIN-DAG: ear %r2, %a1
; CHECK-MAIN-DAG: larl %r1, .LCP{{.*}}
; CHECK-MAIN: ag %r2, 0(%r1)
; CHECK-MAIN: br %r14
  ret ptr@x
}
