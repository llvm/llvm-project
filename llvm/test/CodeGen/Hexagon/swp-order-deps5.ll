; RUN: llc -march=hexagon -hexagon-bit=0 < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 1, %b0 ], [ %v9, %b1 ]
  %v1 = phi i64 [ 0, %b0 ], [ %v10, %b1 ]
  %v2 = load i32, ptr undef, align 4
  %v3 = sub nsw i32 0, %v2
  %v4 = select i1 undef, i32 undef, i32 %v3
  %v5 = sext i32 %v4 to i64
  %v6 = mul nsw i64 %v5, %v5
  %v7 = or i64 %v1, 0
  %v8 = add i64 %v6, %v7
  %v9 = add nuw nsw i32 %v0, 1
  %v10 = and i64 %v8, -4294967296
  %v11 = icmp ult i32 %v9, undef
  br i1 %v11, label %b1, label %b2

b2:                                               ; preds = %b1
  %v12 = sdiv i64 %v8, undef
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv62" }
