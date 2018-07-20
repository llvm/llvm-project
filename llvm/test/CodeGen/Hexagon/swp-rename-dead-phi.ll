; RUN: llc -march=hexagon -fp-contract=fast -enable-pipeliner < %s
; REQUIRES: asserts

; Pipelining can eliminate the need for a Phi if the loop carried use
; is scheduled first. We need to rename register uses of the Phi
; that may occur after the loop.

define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b12

b1:                                               ; preds = %b0
  %v0 = load float, float* undef, align 4
  br i1 undef, label %b2, label %b5

b2:                                               ; preds = %b1
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b3, %b2
  br label %b3

b4:                                               ; preds = %b4, %b2
  br i1 undef, label %b5, label %b4

b5:                                               ; preds = %b4, %b1
  br i1 undef, label %b6, label %b9

b6:                                               ; preds = %b5
  br i1 undef, label %b7, label %b8

b7:                                               ; preds = %b7, %b6
  br label %b7

b8:                                               ; preds = %b8, %b6
  %v1 = phi i32 [ %v7, %b8 ], [ 2, %b6 ]
  %v2 = phi float [ %v6, %b8 ], [ undef, %b6 ]
  %v3 = phi float [ %v2, %b8 ], [ undef, %b6 ]
  %v4 = fmul float undef, %v2
  %v5 = fsub float %v4, %v3
  %v6 = fadd float %v5, undef
  %v7 = add nsw i32 %v1, 1
  %v8 = icmp eq i32 %v7, undef
  br i1 %v8, label %b9, label %b8

b9:                                               ; preds = %b8, %b5
  %v9 = phi float [ undef, %b5 ], [ %v2, %b8 ]
  %v10 = fsub float 0.000000e+00, %v9
  %v11 = fadd float %v10, undef
  %v12 = fmul float undef, %v11
  %v13 = fcmp ugt float %v12, 0.000000e+00
  br i1 %v13, label %b10, label %b11

b10:                                              ; preds = %b9
  br label %b11

b11:                                              ; preds = %b10, %b9
  %v14 = phi float [ undef, %b10 ], [ %v0, %b9 ]
  %v15 = fadd float undef, %v14
  br label %b13

b12:                                              ; preds = %b0
  ret void

b13:                                              ; preds = %b13, %b11
  br label %b13
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
