; Test initial-exec TLS accesses.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=CHECK-MAIN

@x = thread_local(initialexec) global i32 0

; The offset must be loaded from the GOT.  This TLS access model does
; not use literal pool constants.
define i32 *@foo() {
; CHECK-MAIN-LABEL: foo:
; CHECK-MAIN: ear [[HIGH:%r[0-5]]], %a0
; CHECK-MAIN: sllg %r2, [[HIGH]], 32
; CHECK-MAIN-DAG: ear %r2, %a1
; CHECK-MAIN-DAG: larl %r1, x@INDNTPOFF
; CHECK-MAIN: ag %r2, 0(%r1)
; CHECK-MAIN: br %r14
  ret i32 *@x
}
