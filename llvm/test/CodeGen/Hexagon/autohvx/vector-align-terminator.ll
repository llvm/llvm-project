; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: jumpr r31

target triple = "hexagon"

define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0, %b1
  %v0 = phi i32 [ %v9, %b1 ], [ 0, %b0 ]
  %v1 = zext i32 %v0 to i64
  %v2 = getelementptr inbounds float, ptr null, i64 %v1
  store float poison, ptr %v2, align 16
  %v3 = or i32 %v0, 3
  %v4 = zext i32 %v3 to i64
  %v5 = getelementptr inbounds float, ptr null, i64 %v4
  store float poison, ptr %v5, align 4
  %v6 = add nuw nsw i32 %v0, 4
  %v7 = icmp ult i32 %v3, 63
  %v8 = select i1 %v7, i1 true, i1 false
  %v9 = select i1 %v7, i32 %v6, i32 0
  br i1 %v8, label %b1, label %b2, !prof !0

b2:
  ret void
}

attributes #0 = { "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp" }

!0 = !{!"branch_weights", i32 -2147481600, i32 2048}
