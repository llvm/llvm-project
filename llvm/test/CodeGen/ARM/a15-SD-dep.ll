; RUN: llc -O1 -mcpu=cortex-a15 -mtriple=armv7-linux-gnueabi -disable-a15-sd-optimization -verify-machineinstrs < %s  | FileCheck -check-prefix=CHECK-DISABLED %s
; RUN: llc -O1 -mcpu=cortex-a15 -mtriple=armv7-linux-gnueabi -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK-ENABLED %s

; CHECK-ENABLED-LABEL: t1:
; CHECK-DISABLED-LABEL: t1:
define <2 x float> @t1(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED-LABEL: t2:
; CHECK-DISABLED-LABEL: t2:
define <4 x float> @t2(float %g, float %f) {
  ; CHECK-ENABLED: vdup.32 q{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <4 x float> undef, float %f, i32 1
  %i2 = fadd <4 x float> %i1, %i1
  ret <4 x float> %i2
}

; CHECK-ENABLED-LABEL: t3:
; CHECK-DISABLED-LABEL: t3:
define arm_aapcs_vfpcc <2 x float> @t3(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0] 
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED-LABEL: t4:
; CHECK-DISABLED-LABEL: t4:
define <2 x float> @t4(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  br label %b

  ; Block %b has an S-reg as live-in.
b:
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED-LABEL: t5:
; CHECK-DISABLED-LABEL: t5:
define arm_aapcs_vfpcc <4 x float> @t5(<4 x float> %q, float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[0]
  ; CHECK-ENABLED: vadd.f32
  ; CHECK-ENABLED-NEXT: bx lr
  ; CHECK-DISABLED-NOT: vdup
  %i1 = insertelement <4 x float> %q, float %f, i32 1
  %i2 = fadd <4 x float> %i1, %i1
  ret <4 x float> %i2
}

; Test that DPair can be successfully passed as QPR.
; CHECK-ENABLED-LABEL: test_DPair1:
; CHECK-DISABLED-LABEL: test_DPair1:
define void @test_DPair1(i32 %vsout, i8* nocapture %out, float %x, float %y) {
entry:
  %0 = insertelement <4 x float> undef, float %x, i32 1
  %1 = insertelement <4 x float> %0, float %y, i32 0
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[0]
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[1]
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[0]
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[1]
  ; CHECK-DISABLED-NOT: vdup
  switch i32 %vsout, label %sw.epilog [
    i32 1, label %sw.bb
    i32 0, label %sw.bb6
  ]

sw.bb:                                            ; preds = %entry
  %2 = insertelement <4 x float> %1, float 0.000000e+00, i32 0
  br label %sw.bb6

sw.bb6:                                           ; preds = %sw.bb, %entry
  %sum.0 = phi <4 x float> [ %1, %entry ], [ %2, %sw.bb ]
  %3 = extractelement <4 x float> %sum.0, i32 0
  %conv = fptoui float %3 to i8
  store i8 %conv, i8* %out, align 1
  ret void

sw.epilog:                                        ; preds = %entry
  ret void
}

; CHECK-ENABLED-LABEL: test_DPair2:
; CHECK-DISABLED-LABEL: test_DPair2:
define void @test_DPair2(i32 %vsout, i8* nocapture %out, float %x) {
entry:
  %0 = insertelement <4 x float> undef, float %x, i32 0
  ; CHECK-ENABLED: vdup.32 q{{[0-9]*}}, d{{[0-9]*}}[0]
  ; CHECK-DISABLED-NOT: vdup
  switch i32 %vsout, label %sw.epilog [
    i32 1, label %sw.bb
    i32 0, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  %1 = insertelement <4 x float> %0, float 0.000000e+00, i32 0
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %sw.bb
  %sum.0 = phi <4 x float> [ %0, %entry ], [ %1, %sw.bb ]
  %2 = extractelement <4 x float> %sum.0, i32 0
  %conv = fptoui float %2 to i8
  store i8 %conv, i8* %out, align 1
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb1
  ret void
}
