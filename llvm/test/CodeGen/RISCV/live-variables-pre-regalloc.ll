; RUN: llc -mtriple riscv64 -mattr=+d -riscv-enable-live-variables \
; RUN: --stop-after=riscv-live-variables -riscv-liveness-update-kills < %s | FileCheck %s

; RUN: llc -mtriple riscv64 -mattr=+d -riscv-enable-live-variables \
; RUN: -riscv-liveness-update-kills < %s | FileCheck --check-prefix=CHECK-LICM %s

; Issue: #166141 Pessimistic MachineLICM due to missing liveness info.

; Check that live variable analysis correctly marks %41 as kill
; CHECK:  bb.2.if:
; CHECK:    successors: %bb.3(0x80000000)
;
; CHECK:    %40:gpr = LUI target-flags(riscv-hi) %const.0
; CHECK:    %41:fpr64 = FLD killed %40, target-flags(riscv-lo) %const.0 :: (load (s64) from constant-pool)
; CHECK:    %42:fpr64 = nofpexcept FMUL_D killed %2, killed %41, 7, implicit $frm
; CHECK:    FSD killed %42, %1, 0 :: (store (s64) into %ir.lsr.iv1)

; Check that the loop invariant `fld` is hoisted out of the loop.
; CHECK-LICM: # %bb.0:
; CHECK-LICM:        lui     a1, %hi(.LCPI0_0)
; CHECK-LICM:        fld     fa5, %lo(.LCPI0_0)(a1)
; CHECK-LICM:        lui     a1, 2
; CHECK-LICM:        add     a1, a0, a1
; CHECK-LICM:        fmv.d.x fa4, zero
; CHECK-LICM:        j       .LBB0_2
; CHECK-LICM: .LBB0_1:

define void @f(ptr %p) {
entry:
  br label %loop

loop:
  %iv = phi i64 [0, %entry], [%iv.next, %latch]

  %gep = getelementptr double, ptr %p, i64 %iv

  %x = load double, ptr %gep
  %y0 = fmul double %x, %x
  %y1 = fmul double %y0, %y0
  %y2 = fmul double %y1, %y1
  %y3 = fmul double %y2, %y2
  %y4 = fmul double %y3, %y3
  %y5 = fmul double %y4, %y4
  %y6 = fmul double %y5, %y5
  %y7 = fmul double %y6, %y6
  %y8 = fmul double %y7, %y7
  %y9 = fmul double %y8, %y8
  %y10 = fmul double %y9, %y9
  %y11 = fmul double %y10, %y10
  %y12 = fmul double %y11, %y11
  %y13 = fmul double %y12, %y12
  %y14 = fmul double %y13, %y13
  %y15 = fmul double %y14, %y14
  %y16 = fmul double %y15, %y15
  %y17 = fmul double %y16, %y16
  %y18 = fmul double %y17, %y17
  %y19 = fmul double %y18, %y18
  %y20 = fmul double %y19, %y19
  %y21 = fmul double %y20, %y20
  %y22 = fmul double %y21, %y21
  %y23 = fmul double %y22, %y22
  %y24 = fmul double %y23, %y23
  %y25 = fmul double %y24, %y24
  %y26 = fmul double %y25, %y25
  %y27 = fmul double %y26, %y26
  %y28 = fmul double %y27, %y27
  %y29 = fmul double %y28, %y28
  %y30 = fmul double %y29, %y29
  %y31 = fmul double %y30, %y30

  %c = fcmp une double %y31, 0.0
  br i1 %c, label %if, label %latch

if:
  %z = fmul double %y31, 3.14159274101257324218750
  store double %z, ptr %gep
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
