; RUN: llc -O2 -mtriple=hexagon < %s
; REQUIRES: asserts

%0 = type { [2 x [8 x [16 x i8]]], [4 x [16 x ptr]] }
%1 = type { i32, i32, i8, i8, %2, ptr }
%2 = type { i32, i32, ptr, i8, i16, i16, i8 }
%3 = type { i16, i16, %4, i16, i8, i16, %5, i32 }
%4 = type { i32 }
%5 = type { i16, i16 }
%6 = type { ptr }
%7 = type { [16 x i16], [16 x i16] }

; Function Attrs: norecurse nounwind
define void @f0(ptr nocapture %a0) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v6, %b1 ]
  %v1 = getelementptr inbounds %0, ptr %a0, i32 0, i32 1, i32 3, i32 %v0
  %v3 = load i32, ptr %v1, align 4
  store i32 %v3, ptr undef, align 4
  %v4 = getelementptr inbounds %0, ptr %a0, i32 0, i32 1, i32 0, i32 %v0
  store i32 %v3, ptr %v4, align 4
  %v6 = add nuw nsw i32 %v0, 1
  %v7 = icmp eq i32 %v6, 16
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
