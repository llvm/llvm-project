; Test spilling of GPRs.  The tests here assume z10 register pressure,
; without the high words being available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; We need to allocate a 4-byte spill slot, rounded to 8 bytes.  The frame
; size should be exactly 160 + 8 = 168.
define void @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK: aghi %r15, -168
; CHECK-NOT: 160(%r15)
; CHECK: st [[REGISTER:%r[0-9]+]], 164(%r15)
; CHECK-NOT: 160(%r15)
; CHECK: l [[REGISTER]], 164(%r15)
; CHECK-NOT: 160(%r15)
; CHECK: lmg %r6, %r15, 216(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l3 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l6 = load volatile i32, i32 *%ptr
  %l7 = load volatile i32, i32 *%ptr
  %l8 = load volatile i32, i32 *%ptr
  %l9 = load volatile i32, i32 *%ptr
  %l10 = load volatile i32, i32 *%ptr
  %l11 = load volatile i32, i32 *%ptr
  %l12 = load volatile i32, i32 *%ptr
  %l13 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %lx = load volatile i32, i32 *%ptr
  store volatile i32 %lx, i32 *%ptr
  store volatile i32 %l14, i32 *%ptr
  store volatile i32 %l13, i32 *%ptr
  store volatile i32 %l12, i32 *%ptr
  store volatile i32 %l11, i32 *%ptr
  store volatile i32 %l10, i32 *%ptr
  store volatile i32 %l9, i32 *%ptr
  store volatile i32 %l8, i32 *%ptr
  store volatile i32 %l7, i32 *%ptr
  store volatile i32 %l6, i32 *%ptr
  store volatile i32 %l5, i32 *%ptr
  store volatile i32 %l4, i32 *%ptr
  store volatile i32 %l3, i32 *%ptr
  store volatile i32 %l1, i32 *%ptr
  store volatile i32 %l0, i32 *%ptr
  ret void
}

; Same for i64, except that the full spill slot is used.
define void @f2(i64 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK: aghi %r15, -168
; CHECK: stg [[REGISTER:%r[0-9]+]], 160(%r15)
; CHECK: lg [[REGISTER]], 160(%r15)
; CHECK: lmg %r6, %r15, 216(%r15)
; CHECK: br %r14
  %l0 = load volatile i64, i64 *%ptr
  %l1 = load volatile i64, i64 *%ptr
  %l3 = load volatile i64, i64 *%ptr
  %l4 = load volatile i64, i64 *%ptr
  %l5 = load volatile i64, i64 *%ptr
  %l6 = load volatile i64, i64 *%ptr
  %l7 = load volatile i64, i64 *%ptr
  %l8 = load volatile i64, i64 *%ptr
  %l9 = load volatile i64, i64 *%ptr
  %l10 = load volatile i64, i64 *%ptr
  %l11 = load volatile i64, i64 *%ptr
  %l12 = load volatile i64, i64 *%ptr
  %l13 = load volatile i64, i64 *%ptr
  %l14 = load volatile i64, i64 *%ptr
  %lx = load volatile i64, i64 *%ptr
  store volatile i64 %lx, i64 *%ptr
  store volatile i64 %l14, i64 *%ptr
  store volatile i64 %l13, i64 *%ptr
  store volatile i64 %l12, i64 *%ptr
  store volatile i64 %l11, i64 *%ptr
  store volatile i64 %l10, i64 *%ptr
  store volatile i64 %l9, i64 *%ptr
  store volatile i64 %l8, i64 *%ptr
  store volatile i64 %l7, i64 *%ptr
  store volatile i64 %l6, i64 *%ptr
  store volatile i64 %l5, i64 *%ptr
  store volatile i64 %l4, i64 *%ptr
  store volatile i64 %l3, i64 *%ptr
  store volatile i64 %l1, i64 *%ptr
  store volatile i64 %l0, i64 *%ptr
  ret void
}
