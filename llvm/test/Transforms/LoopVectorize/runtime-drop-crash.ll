; RUN: opt -passes=loop-vectorize -force-vector-width=4 %s | FileCheck %s

%struct.foo = type { [400 x double] }

; Make sure we do not crash when dropping runtime checks.

; CHECK-NOT: vector.body

define void @barney(ptr %ptr) {
entry:
  br label %loop

loop:
  %tmp3 = phi i64 [ 0, %entry ], [ %tmp18, %loop ]
  %tmp4 = getelementptr inbounds %struct.foo, ptr %ptr, i64 undef
  store i64 0, ptr %tmp4, align 8
  %tmp8 = add i64 1, %tmp3
  %tmp10 = getelementptr inbounds %struct.foo, ptr %ptr, i64 %tmp8
  store i64 1, ptr %tmp10, align 8
  %tmp14 = add i64 undef, %tmp3
  %tmp16 = getelementptr inbounds %struct.foo, ptr %ptr, i64 %tmp14
  store i64 2, ptr %tmp16, align 8
  %tmp18 = add nuw nsw i64 %tmp3, 4
  %c = icmp ult i64 %tmp18, 400
  br i1 %c, label %exit, label %loop

exit:
  ret void
}
