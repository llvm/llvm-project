; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s

define void @test_cmpxchg_weak(i32 *%addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak:

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  %oldval = extractvalue { i32, i1 } %pair, 0
; CHECK-NEXT: @ %bb.0:                                @ %cmpxchg.start
; CHECK-NEXT:     ldrex   r3, [r0]
; CHECK-NEXT:     cmp     r3, r1
; CHECK-NEXT:     beq     LBB0_2
; CHECK-NEXT: @ %bb.1:                                @ %cmpxchg.nostore
; CHECK-NEXT:     clrex
; CHECK-NEXT:     b       LBB0_3
; CHECK-NEXT: LBB0_2:                                 @ %cmpxchg.fencedstore
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     strex   r1, r2, [r0]
; CHECK-NEXT:     cmp     r1, #0
; CHECK-NEXT:     beq     LBB0_4
; CHECK-NEXT: LBB0_3:                                 @ %cmpxchg.end
; CHECK-NEXT:     str     r3, [r0]
; CHECK-NEXT:     bx      lr
; CHECK-NEXT: LBB0_4:                                 @ %cmpxchg.success
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     str     r3, [r0]
; CHECK-NEXT:     bx      lr


  store i32 %oldval, i32* %addr
  ret void
}


define i1 @test_cmpxchg_weak_to_bool(i32, i32 *%addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_weak_to_bool:

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  %success = extractvalue { i32, i1 } %pair, 1

; CHECK-NEXT: @ %bb.0:                                @ %cmpxchg.start
; CHECK-NEXT:     ldrex   r0, [r1]
; CHECK-NEXT:     cmp     r0, r2
; CHECK-NEXT:     beq     LBB1_2
; CHECK-NEXT: @ %bb.1:                                @ %cmpxchg.nostore
; CHECK-NEXT:     mov     r0, #0
; CHECK-NEXT:     clrex
; CHECK-NEXT:     bx      lr
; CHECK-NEXT: LBB1_2:                                 @ %cmpxchg.fencedstore
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     mov     r0, #0
; CHECK-NEXT:     strex   r2, r3, [r1]
; CHECK-NEXT:     cmp     r2, #0
; CHECK-NEXT:     bxne    lr
; CHECK-NEXT:     mov     r0, #1
; CHECK-NEXT:     dmb     ish
; CHECK-NEXT:     bx      lr
                    

  ret i1 %success
}
