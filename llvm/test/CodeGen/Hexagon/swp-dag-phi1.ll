; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; This test check that a dependence is created between a Phi and it's uses.
; An assert occurs if the Phi dependences are not correct.

define void @f0(ptr nocapture %a0, i32 %a1) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi float [ %v1, %b1 ], [ undef, %b0 ]
  %v1 = phi float [ %v13, %b1 ], [ undef, %b0 ]
  %v2 = phi ptr [ null, %b1 ], [ %a0, %b0 ]
  %v3 = phi i32 [ %v14, %b1 ], [ 0, %b0 ]
  %v4 = phi float [ %v5, %b1 ], [ undef, %b0 ]
  %v5 = load float, ptr %v2, align 4
  %v6 = fmul float %v1, 0x3FFFA98000000000
  %v7 = fmul float %v0, 0xBFEF550000000000
  %v8 = fadd float %v6, %v7
  %v9 = fmul float %v5, 0x3FEFAA0000000000
  %v10 = fadd float %v8, %v9
  %v11 = fmul float %v4, 0xBFFFAA0000000000
  %v12 = fadd float %v11, %v10
  %v13 = fadd float undef, %v12
  store float %v13, ptr %v2, align 4
  %v14 = add nsw i32 %v3, 1
  %v15 = icmp eq i32 %v14, %a1
  br i1 %v15, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
