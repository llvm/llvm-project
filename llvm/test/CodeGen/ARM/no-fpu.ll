; RUN: llc < %s -mtriple=armv7-none-gnueabi -mattr=-neon,-fpregs | FileCheck --check-prefix=NONEON-NOVFP %s
; RUN: llc < %s -mtriple=armv7-none-gnueabi -mattr=-neon | FileCheck --check-prefix=NONEON %s
; RUN: llc < %s -mtriple=armv7-none-gnueabi -mattr=-fpregs | FileCheck --check-prefix=NOVFP %s
; RUN: llc < %s -mtriple=armv7-none-gnueabi -mattr=-neon,+vfp2 | FileCheck --check-prefix=NONEON-VFP %s

; Check no NEON instructions are selected when feature is disabled.
define void @neonop(ptr nocapture readonly %a, ptr nocapture %b) #0 {
  %wide.load = load <2 x i64>, ptr %a, align 8
  ; NONEON-NOVFP-NOT: vld1.64
  ; NONEON-NOT: vld1.64
  %add = add <2 x i64> %wide.load, %wide.load
  ; NONEON-NOVFP-NOT: vadd.i64
  ; NONEON-NOT: vadd.i64
  store <2 x i64> %add, ptr %b, align 8
  ; NONEON-NOVFP-NOT: vst1.64
  ; NONEON-NOT: vst1.64
  ret void
}

; Likewise with VFP instructions.
define double @fpmult(double %a, double %b) {
  %res = fmul double %a, %b
  ; NONEON-NOVFP-NOT: vmov
  ; NONEON-NOVFP-NOT: vmul.f64
  ; NOVFP-NOT: vmov
  ; NOVFP-NOT: vmul.f64
  ; NONEON-VFP: vmov
  ; NONEON-VFP: vmul.f64
  ret double %res
}

