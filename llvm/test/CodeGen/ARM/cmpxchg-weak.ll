; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s

define void @test_cmpxchg_weak(ptr %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak:
; CHECK-NEXT: %bb.0:
; CHECK-NEXT:     ldrex   [[LOADED:r[0-9]+]], [r0]
; CHECK-NEXT:     cmp     [[LOADED]], r1
; CHECK-NEXT:     bne     [[LDFAILBB:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: %bb.1:
; CHECK-NEXT:     dmb ish
; CHECK-NEXT:     strex   [[SUCCESS:r[0-9]+]], r2, [r0]
; CHECK-NEXT:     cmp     [[SUCCESS]], #0
; CHECK-NEXT:     bne     [[FAILBB:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: %bb.2:
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     str     r3, [r0]
; CHECK-NEXT:     bx      lr
; CHECK-NEXT: [[LDFAILBB]]:
; CHECK-NEXT:     clrex
; CHECK-NEXT: [[FAILBB]]:
; CHECK-NEXT:     str     r3, [r0]
; CHECK-NEXT:     bx      lr
;
  %pair = cmpxchg weak ptr %addr, i32 %desired, i32 %new seq_cst monotonic
  %oldval = extractvalue { i32, i1 } %pair, 0
  store i32 %oldval, ptr %addr
  ret void
}

define i1 @test_cmpxchg_weak_to_bool(i32, ptr %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak_to_bool:
; CHECK-NEXT: %bb.0:
; CHECK-NEXT:     ldrex   [[LOADED:r[0-9]+]], [r1]
; CHECK-NEXT:     cmp     [[LOADED]], r2
; CHECK-NEXT:     bne     [[LDFAILBB:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: %bb.1:
; CHECK-NEXT:     dmb ish
; CHECK-NEXT:     strex   [[SUCCESS:r[0-9]+]], r3, [r1]
; CHECK-NEXT:     cmp     [[SUCCESS]], #0
; CHECK-NEXT:     bne     [[FAILBB:LBB[0-9]+_[0-9]+]] 
; CHECK-NEXT: %bb.2:
; CHECK-NEXT:     mov     r0, #1
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     bx      lr
; CHECK-NEXT: [[LDFAILBB]]:
; CHECK-NEXT:     clrex
; CHECK-NEXT: [[FAILBB]]:
; CHECK-NEXT:     mov     r0, #0
; CHECK-NEXT:     bx      lr
;

  %pair = cmpxchg weak ptr %addr, i32 %desired, i32 %new seq_cst monotonic
  %success = extractvalue { i32, i1 } %pair, 1
  ret i1 %success
}
