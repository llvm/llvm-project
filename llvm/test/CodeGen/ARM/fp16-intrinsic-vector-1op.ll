; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+v8.2a,+fullfp16,+neon  -float-abi=hard   | FileCheck %s --check-prefixes=CHECK,CHECK-HARD
; RUN: llc < %s -mtriple=armeb-none-eabi -mattr=+v8.2a,+fullfp16,+neon  -float-abi=hard   | FileCheck %s --check-prefixes=CHECK,CHECK-HARD-BE
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+v8.2a,+fullfp16,+neon  | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP
; RUN: llc < %s -mtriple=armeb-none-eabi -mattr=+v8.2a,+fullfp16,+neon  | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP-BE

declare <8 x half> @llvm.fabs.v8f16(<8 x half>)

define dso_local <8 x half> @t_vabsq_f16(<8 x half> %a) {
; CHECK-LABEL:      t_vabsq_f16:

; CHECK-HARD:         vabs.f16  q0, q0
; CHECK-HARD-NEXT:    bx  lr

; CHECK-HARD-BE:      vrev64.16 [[Q8:q[0-9]+]], q0
; CHECK-HARD-BE-NEXT: vabs.f16  [[Q8]], [[Q8]]
; CHECK-HARD-BE-NEXT: vrev64.16 q0, [[Q8]]
; CHECK-HARD-BE-NEXT: bx  lr

; CHECK-SOFTFP:       vmov  d{{.*}}, r2, r3
; CHECK-SOFTFP:       vmov  d{{.*}}, r0, r1
; CHECK-SOFTFP:       vabs.f16  q{{.*}}, q{{.*}}
; CHECK-SOFTFP:       vmov  r0, r1, d{{.*}}
; CHECK-SOFTFP:       vmov  r2, r3, d{{.*}}
; CHECK-SOFTFP:       bx  lr

; CHECK-SOFTFP-BE:    vmov  [[D17:d[0-9]+]], r3, r2
; CHECK-SOFTFP-BE:    vmov  [[D16:d[0-9]+]], r1, r0
; CHECK-SOFTFP-BE:    vrev64.16 [[Q8:q[0-9]+]], [[Q8]]
; CHECK-SOFTFP-BE:    vabs.f16  [[Q8]], [[Q8]]
; CHECK-SOFTFP-BE:    vrev64.16 [[Q8]], [[Q8]]
; CHECK-SOFTFP-BE:    vmov  r1, r0, [[D16]]
; CHECK-SOFTFP-BE:    vmov  r3, r2, [[D17]]
; CHECK-SOFTFP-BE:    bx  lr

entry:
  %vabs1.i = tail call <8 x half> @llvm.fabs.v8f16(<8 x half> %a) #3
  ret <8 x half> %vabs1.i
}

