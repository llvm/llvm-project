; RUN: opt -passes=loop-vectorize -force-vector-width=4 %s | FileCheck %s

%struct.foo = type { [400 x double] }

; Make sure we do not crash when dropping runtime checks.

; CHECK-NOT: vector.body

define void @barney(%struct.foo* %ptr) {
entry:
  br label %loop

loop:
  %tmp3 = phi i64 [ 0, %entry ], [ %tmp18, %loop ]
  %tmp4 = getelementptr inbounds %struct.foo, %struct.foo* %ptr, i64 undef
  %tmp5 = bitcast %struct.foo* %tmp4 to i64*
  store i64 0, i64* %tmp5, align 8
  %tmp8 = add i64 1, %tmp3
  %tmp10 = getelementptr inbounds %struct.foo, %struct.foo* %ptr, i64 %tmp8
  %tmp11 = bitcast %struct.foo* %tmp10 to i64*
  store i64 1, i64* %tmp11, align 8
  %tmp14 = add i64 undef, %tmp3
  %tmp16 = getelementptr inbounds %struct.foo, %struct.foo* %ptr, i64 %tmp14
  %tmp17 = bitcast %struct.foo* %tmp16 to i64*
  store i64 2, i64* %tmp17, align 8
  %tmp18 = add nuw nsw i64 %tmp3, 4
  %c = icmp ult i64 %tmp18, 400
  br i1 %c, label %exit, label %loop

exit:
  ret void
}
