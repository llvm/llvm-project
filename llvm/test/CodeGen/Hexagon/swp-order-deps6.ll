; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

%s.0 = type { i64 }

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #0

define void @f0(%s.0* noalias nocapture %a0, i32 %a1) local_unnamed_addr {
b0:
  %v0 = call i64 @llvm.hexagon.A2.combinew(i32 %a1, i32 %a1)
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ 0, %b0 ], [ %v6, %b1 ]
  %v2 = mul nuw nsw i32 %v1, 13
  %v3 = getelementptr inbounds %s.0, %s.0* %a0, i32 %v2, i32 0
  %v4 = load i64, i64* %v3, align 8
  %v5 = add nsw i64 %v4, %v0
  store i64 %v5, i64* %v3, align 8
  %v6 = add nuw nsw i32 %v1, 1
  %v7 = icmp eq i32 %v6, 12
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind readnone }
