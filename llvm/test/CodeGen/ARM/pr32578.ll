; RUN: llc -o - %s | FileCheck %s
target triple = "armv7"

; CHECK-LABEL: func:
; CHECK: push {r11, lr}
; CHECK: vpush {d8}
; CEHCK: b .LBB0_2
define arm_aapcscc double @func() {
  br label %tailrecurse

tailrecurse:
  %v0 = load i16, i16* undef, align 8
  %cond36.i = icmp eq i16 %v0, 3
  br i1 %cond36.i, label %sw.bb.i, label %sw.epilog.i

sw.bb.i:
  %v1 = load double, double* undef, align 8
  %call21.i = tail call arm_aapcscc double @func()
  %mul.i = fmul double %v1, %call21.i
  ret double %mul.i

sw.epilog.i:
  tail call arm_aapcscc void @_ZNK10shared_ptrdeEv()
  br label %tailrecurse
}

declare arm_aapcscc void @_ZNK10shared_ptrdeEv() local_unnamed_addr
