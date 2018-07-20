; RUN: llc -march=hexagon -enable-pipeliner -fp-contract=fast < %s
; REQUIRES: asserts

; Test that the code which reuses existing Phis works when the Phis are used
; in multiple stages. In this case, one can be reused, but the other must be
; generated.

; Function Attrs: nounwind
define void @f0(i32 %a0) #0 {
b0:
  %v0 = icmp sgt i32 %a0, 0
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v11, %b1 ], [ 0, %b0 ]
  %v2 = phi float [ %v4, %b1 ], [ undef, %b0 ]
  %v3 = phi float [ %v2, %b1 ], [ undef, %b0 ]
  %v4 = load float, float* undef, align 4
  %v5 = fmul float %v4, 0x3FEFAA0000000000
  %v6 = fadd float undef, %v5
  %v7 = fmul float %v2, 0xBFFFAA0000000000
  %v8 = fadd float %v7, %v6
  %v9 = fmul float %v3, 0x3FEFAA0000000000
  %v10 = fadd float %v9, %v8
  store float %v10, float* undef, align 4
  %v11 = add nsw i32 %v1, 1
  %v12 = icmp eq i32 %v11, %a0
  br i1 %v12, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
