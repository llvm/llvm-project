; RUN: llc -mtriple=hexagon -fp-contract=fast -enable-pipeliner < %s
; REQUIRES: asserts

; A Phi that depends on another Phi is loop carried.

define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b1, label %b2

b2:                                               ; preds = %b1
  br i1 undef, label %b3, label %b8

b3:                                               ; preds = %b2
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b4, %b3
  %v0 = phi i32 [ %v5, %b4 ], [ 2, %b3 ]
  %v1 = phi float [ %v4, %b4 ], [ undef, %b3 ]
  %v2 = phi float [ %v1, %b4 ], [ undef, %b3 ]
  %v3 = fsub float 0.000000e+00, %v2
  %v4 = fadd float %v3, undef
  %v5 = add nsw i32 %v0, 1
  %v6 = icmp eq i32 %v5, undef
  br i1 %v6, label %b5, label %b4

b5:                                               ; preds = %b4, %b3
  %v7 = phi float [ undef, %b3 ], [ %v1, %b4 ]
  br i1 false, label %b6, label %b7

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b6, %b5
  br label %b9

b8:                                               ; preds = %b2
  ret void

b9:                                               ; preds = %b9, %b7
  br label %b9
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
