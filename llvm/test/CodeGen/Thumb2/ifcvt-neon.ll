; RUN: llc < %s -march=thumb -mcpu=cortex-a8 | FileCheck %s
; rdar://7368193

@a = common global float 0.000000e+00             ; <float*> [#uses=2]
@b = common global float 0.000000e+00             ; <float*> [#uses=1]

define float @t(i32 %c) nounwind {
entry:
  %0 = icmp sgt i32 %c, 1                         ; <i1> [#uses=1]
  %1 = load float* @a, align 4                    ; <float> [#uses=2]
  %2 = load float* @b, align 4                    ; <float> [#uses=2]
  br i1 %0, label %bb, label %bb1

bb:                                               ; preds = %entry
; CHECK:      ite lt
; CHECK:      vsublt.f32
; CHECK-NEXT: vaddge.f32
  %3 = fadd float %1, %2                          ; <float> [#uses=1]
  br label %bb2

bb1:                                              ; preds = %entry
  %4 = fsub float %1, %2                          ; <float> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %storemerge = phi float [ %4, %bb1 ], [ %3, %bb ] ; <float> [#uses=2]
  store float %storemerge, float* @a
  ret float %storemerge
}
