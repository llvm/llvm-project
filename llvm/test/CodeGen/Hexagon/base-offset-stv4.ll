; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b4

b1:                                               ; preds = %b0
  %v0 = load ptr, ptr undef, align 4
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 13, %b1 ], [ %v5, %b2 ]
  %v2 = getelementptr inbounds i16, ptr %v0, i32 %v1
  %v3 = add nsw i32 0, %v1
  %v4 = getelementptr inbounds i16, ptr %v0, i32 %v3
  store i16 0, ptr %v4, align 2
  store i16 0, ptr %v2, align 2
  %v5 = add i32 %v1, 1
  %v6 = icmp eq i32 %v5, 26
  br i1 %v6, label %b3, label %b2

b3:                                               ; preds = %b3, %b2
  br i1 undef, label %b4, label %b3

b4:                                               ; preds = %b3, %b0
  ret void
}

attributes #0 = { nounwind }
