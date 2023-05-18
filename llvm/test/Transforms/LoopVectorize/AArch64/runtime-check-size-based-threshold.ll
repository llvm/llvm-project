; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-iphoneos -vectorizer-min-trip-count=8 \
; RUN:   -enable-epilogue-vectorization=false -S %s | FileCheck --check-prefixes=CHECK,DEFAULT %s
; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-iphoneos -vectorizer-min-trip-count=8 \
; RUN:   -enable-epilogue-vectorization=false -vectorize-memory-check-threshold=1 -S %s | FileCheck --check-prefixes=CHECK,THRESHOLD %s

; Tests for loops with large numbers of runtime checks. Check that loops are
; vectorized, if the loop trip counts are large and the impact of the runtime
; checks is very small compared to the expected loop runtimes.


; The trip count in the loop in this function is too to warrant large runtime checks.
; CHECK-LABEL: define {{.*}} @test_tc_too_small
; CHECK-NOT: vector.memcheck
; CHECK-NOT: vector.body
define void @test_tc_too_small(ptr %ptr.1, ptr %ptr.2, ptr %ptr.3, ptr %ptr.4, i64 %off.1, i64 %off.2) {
entry:
  br label %loop

loop:                                             ; preds = %bb54, %bb37
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i16, ptr %ptr.1, i64 %iv
  %lv.1 = load i16, ptr %gep.1, align 2
  %ext.1 = sext i16 %lv.1 to i32
  %gep.2 = getelementptr inbounds i16, ptr %ptr.2, i64 %iv
  %lv.2 = load i16, ptr %gep.2, align 2
  %ext.2 = sext i16 %lv.2 to i32
  %gep.off.1 = getelementptr inbounds i16, ptr %gep.2, i64 %off.1
  %lv.3 = load i16, ptr %gep.off.1, align 2
  %ext.3 = sext i16 %lv.3 to i32
  %gep.off.2 = getelementptr inbounds i16, ptr %gep.2, i64 %off.2
  %lv.4 = load i16, ptr %gep.off.2, align 2
  %ext.4 = sext i16 %lv.4 to i32
  %tmp62 = mul nsw i32 %ext.2, 11
  %tmp66 = mul nsw i32 %ext.3, -4
  %tmp70 = add nsw i32 %tmp62, 4
  %tmp71 = add nsw i32 %tmp70, %tmp66
  %tmp72 = add nsw i32 %tmp71, %ext.4
  %tmp73 = lshr i32 %tmp72, 3
  %tmp74 = add nsw i32 %tmp73, %ext.1
  %tmp75 = lshr i32 %tmp74, 1
  %tmp76 = mul nsw i32 %ext.2, 5
  %tmp77 = shl nsw i32 %ext.3, 2
  %tmp78 = add nsw i32 %tmp76, 4
  %tmp79 = add nsw i32 %tmp78, %tmp77
  %tmp80 = sub nsw i32 %tmp79, %ext.4
  %tmp81 = lshr i32 %tmp80, 3
  %tmp82 = sub nsw i32 %tmp81, %ext.1
  %tmp83 = lshr i32 %tmp82, 1
  %trunc.1 = trunc i32 %tmp75 to i16
  %gep.3 = getelementptr inbounds i16, ptr %ptr.3, i64 %iv
  store i16 %trunc.1, ptr %gep.3, align 2
  %trunc.2 = trunc i32 %tmp83 to i16
  %gep.4 = getelementptr inbounds i16, ptr %ptr.4, i64 %iv
  store i16 %trunc.2, ptr %gep.4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp ult i64 %iv, 10
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The trip count in the loop in this function high enough to warrant large runtime checks.
; CHECK-LABEL: define {{.*}} @test_tc_big_enough
; DEFAULT: vector.memcheck
; DEFAULT: vector.body
; THRESHOLD-NOT: vector.memcheck
; THRESHOLD-NOT: vector.body
;
define void @test_tc_big_enough(ptr %ptr.1, ptr %ptr.2, ptr %ptr.3, ptr %ptr.4, i64 %off.1, i64 %off.2) {
entry:
  br label %loop

loop:                                             ; preds = %bb54, %bb37
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i16, ptr %ptr.1, i64 %iv
  %lv.1 = load i16, ptr %gep.1, align 2
  %ext.1 = sext i16 %lv.1 to i32
  %gep.2 = getelementptr inbounds i16, ptr %ptr.2, i64 %iv
  %lv.2 = load i16, ptr %gep.2, align 2
  %ext.2 = sext i16 %lv.2 to i32
  %gep.off.1 = getelementptr inbounds i16, ptr %gep.2, i64 %off.1
  %lv.3 = load i16, ptr %gep.off.1, align 2
  %ext.3 = sext i16 %lv.3 to i32
  %gep.off.2 = getelementptr inbounds i16, ptr %gep.2, i64 %off.2
  %lv.4 = load i16, ptr %gep.off.2, align 2
  %ext.4 = sext i16 %lv.4 to i32
  %tmp62 = mul nsw i32 %ext.2, 11
  %tmp66 = mul nsw i32 %ext.3, -4
  %tmp70 = add nsw i32 %tmp62, 4
  %tmp71 = add nsw i32 %tmp70, %tmp66
  %tmp72 = add nsw i32 %tmp71, %ext.4
  %tmp73 = lshr i32 %tmp72, 3
  %tmp74 = add nsw i32 %tmp73, %ext.1
  %tmp75 = lshr i32 %tmp74, 1
  %tmp76 = mul nsw i32 %ext.2, 5
  %tmp77 = shl nsw i32 %ext.3, 2
  %tmp78 = add nsw i32 %tmp76, 4
  %tmp79 = add nsw i32 %tmp78, %tmp77
  %tmp80 = sub nsw i32 %tmp79, %ext.4
  %tmp81 = lshr i32 %tmp80, 3
  %tmp82 = sub nsw i32 %tmp81, %ext.1
  %tmp83 = lshr i32 %tmp82, 1
  %trunc.1 = trunc i32 %tmp75 to i16
  %gep.3 = getelementptr inbounds i16, ptr %ptr.3, i64 %iv
  store i16 %trunc.1, ptr %gep.3, align 2
  %trunc.2 = trunc i32 %tmp83 to i16
  %gep.4 = getelementptr inbounds i16, ptr %ptr.4, i64 %iv
  store i16 %trunc.2, ptr %gep.4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp ult i64 %iv, 500
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_tc_unknown(ptr %ptr.1, ptr %ptr.2, ptr %ptr.3, ptr %ptr.4, i64 %off.1, i64 %off.2, i64 %N) {
; CHECK-LABEL: define void @test_tc_unknown
; DEFAULT:       [[ADD:%.+]] = add i64 %N, 1
; DEFAULT-NEXT:  [[C:%.+]] = icmp ult i64 [[ADD]], 16
; DEFAULT-NEXT:  br i1 [[C]], label %scalar.ph, label %vector.memcheck
; THRESHOLD-NOT: vector.memcheck
; THRESHOLD-NOT: vector.body
;
entry:
  br label %loop

loop:                                             ; preds = %bb54, %bb37
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i16, ptr %ptr.1, i64 %iv
  %lv.1 = load i16, ptr %gep.1, align 2
  %ext.1 = sext i16 %lv.1 to i32
  %gep.2 = getelementptr inbounds i16, ptr %ptr.2, i64 %iv
  %lv.2 = load i16, ptr %gep.2, align 2
  %ext.2 = sext i16 %lv.2 to i32
  %gep.off.1 = getelementptr inbounds i16, ptr %gep.2, i64 %off.1
  %lv.3 = load i16, ptr %gep.off.1, align 2
  %ext.3 = sext i16 %lv.3 to i32
  %gep.off.2 = getelementptr inbounds i16, ptr %gep.2, i64 %off.2
  %lv.4 = load i16, ptr %gep.off.2, align 2
  %ext.4 = sext i16 %lv.4 to i32
  %tmp62 = mul nsw i32 %ext.2, 11
  %tmp66 = mul nsw i32 %ext.3, -4
  %tmp70 = add nsw i32 %tmp62, 4
  %tmp71 = add nsw i32 %tmp70, %tmp66
  %tmp72 = add nsw i32 %tmp71, %ext.4
  %tmp73 = lshr i32 %tmp72, 3
  %tmp74 = add nsw i32 %tmp73, %ext.1
  %tmp75 = lshr i32 %tmp74, 1
  %tmp76 = mul nsw i32 %ext.2, 5
  %tmp77 = shl nsw i32 %ext.3, 2
  %tmp78 = add nsw i32 %tmp76, 4
  %tmp79 = add nsw i32 %tmp78, %tmp77
  %tmp80 = sub nsw i32 %tmp79, %ext.4
  %tmp81 = lshr i32 %tmp80, 3
  %tmp82 = sub nsw i32 %tmp81, %ext.1
  %tmp83 = lshr i32 %tmp82, 1
  %trunc.1 = trunc i32 %tmp75 to i16
  %gep.3 = getelementptr inbounds i16, ptr %ptr.3, i64 %iv
  store i16 %trunc.1, ptr %gep.3, align 2
  %trunc.2 = trunc i32 %tmp83 to i16
  %gep.4 = getelementptr inbounds i16, ptr %ptr.4, i64 %iv
  store i16 %trunc.2, ptr %gep.4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp ult i64 %iv, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
