; RUN: llc < %s -mtriple=aarch64-unknown-unknown | FileCheck %s

define i1 @andn_cmp(i32 %x, i32 %y) {
; CHECK-LABEL: andn_cmp:
; CHECK:       // BB#0:
; CHECK-NEXT:    bics wzr, w1, w0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
;
  %notx = xor i32 %x, -1
  %and = and i32 %notx, %y
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
}

define i1 @and_cmp(i32 %x, i32 %y) {
; CHECK-LABEL: and_cmp:
; CHECK:       // BB#0:
; CHECK-NEXT:    bics wzr, w1, w0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
;
  %and = and i32 %x, %y
  %cmp = icmp eq i32 %and, %y
  ret i1 %cmp
}

define i1 @and_cmp_const(i32 %x) {
; CHECK-LABEL: and_cmp_const:
; CHECK:       // BB#0:
; CHECK-NEXT:    mov w8, #43
; CHECK-NEXT:    bics wzr, w8, w0
; CHECK-NEXT:    cset w0, eq
; CHECK-NEXT:    ret
;
  %and = and i32 %x, 43
  %cmp = icmp eq i32 %and, 43
  ret i1 %cmp
}

