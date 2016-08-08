; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin | FileCheck %s

; TODO: These could use 'andc'.

define i1 @and_cmp1(i32 %x, i32 %y) {
; CHECK-LABEL: and_cmp1:
; CHECK:       ; BB#0:
; CHECK-NEXT:    and r2, r3, r4
; CHECK-NEXT:    li r3, 1
; CHECK-NEXT:    cmpw cr0, r2, r4
; CHECK-NEXT:    bclr 12, 2, 0
; CHECK-NEXT:  ; BB#1:
; CHECK-NEXT:    li r3, 0
; CHECK-NEXT:    blr
;
  %and = and i32 %x, %y
  %cmp = icmp eq i32 %and, %y
  ret i1 %cmp
}

define i1 @and_cmp_const(i32 %x) {
; CHECK-LABEL: and_cmp_const:
; CHECK:       ; BB#0:
; CHECK-NEXT:    andi. r2, r3, 43
; CHECK-NEXT:    li r3, 1
; CHECK-NEXT:    cmpwi r2, 43
; CHECK-NEXT:    bclr 12, 2, 0
; CHECK-NEXT:  ; BB#1:
; CHECK-NEXT:    li r3, 0
; CHECK-NEXT:    blr
;
  %and = and i32 %x, 43
  %cmp = icmp eq i32 %and, 43
  ret i1 %cmp
}

