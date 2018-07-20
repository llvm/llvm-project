; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

%s.0 = type { i64 }

define i64 @f0(%s.0* nocapture %a0, i32 %a1) {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v6, %b1 ], [ 0, %b0 ]
  %v1 = phi i64 [ %v5, %b1 ], [ undef, %b0 ]
  %v2 = load i16, i16* undef, align 2
  %v3 = zext i16 %v2 to i64
  %v4 = and i64 %v1, -4294967296
  %v5 = or i64 %v3, %v4
  %v6 = add nsw i32 %v0, 1
  %v7 = icmp eq i32 %v6, %a1
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  %v8 = phi i64 [ undef, %b0 ], [ %v5, %b1 ]
  ret i64 %v8
}
