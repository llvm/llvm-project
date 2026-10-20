; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that the compiler doesn't assert because the live interval information
; isn't updated correctly during the Hexagon Expand Condsets pass. The pass
; wasn't updating the information when converting a mux with the same operands
; into a copy. When this occurs, the pass needs to update the liveness
; information for the predicate register, which is removed.

define void @f0(i32 %a0) unnamed_addr {
b0:
  %v0 = or i32 undef, %a0
  %v1 = or i32 undef, %v0
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v2 = phi i32 [ %v9, %b3 ], [ 0, %b0 ]
  %v3 = phi i32 [ 0, %b3 ], [ %v1, %b0 ]
  %v4 = srem i32 %v2, 4
  %v5 = icmp eq i32 %v4, 0
  %v6 = select i1 %v5, i32 %v1, i32 %v3
  %v7 = shl i32 %v6, 8
  %v8 = add i32 0, %v7
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  store i32 %v8, ptr undef, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v9 = add nuw nsw i32 %v2, 1
  %v10 = icmp slt i32 %v9, undef
  br i1 %v10, label %b1, label %b4

b4:                                               ; preds = %b3
  unreachable
}
