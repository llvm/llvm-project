; RUN: opt -passes=loop-vectorize -force-vector-width=2 -S %s | FileCheck %s

; Tests with alias sets that contain points with uncomputable bounds because
; they include %offset.1, which is loaded in each loop iteration.

; Alias set with uncomputable bounds contains a single load. We do not need
; runtime checks for that group and it should not block vectorization.
define void @test1_uncomputable_bounds_single_load(ptr noalias %ptr.1, ptr noalias %ptr.2, ptr noalias %ptr.3, i64 %N, i64 %X) {
; CHECK-LABEL: define void @test1_uncomputable_bounds_single_load
; CHECK:       vector.body
; CHECK:         ret void

entry:
  %cond = icmp sgt i64 %N, 0
  br i1 %cond, label %ph, label %exit

ph:
  br label %loop

loop:
  %iv = phi i64 [ 0, %ph ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i32, ptr %ptr.3, i64 %iv
  %offset.1 = load i32, ptr %gep.1, align 4
  %gep.2 = getelementptr inbounds i32, ptr %ptr.2, i32 %offset.1
  %lv = load i32, ptr %gep.2, align 4
  %gep.3 = getelementptr inbounds i32, ptr %ptr.1, i64 %iv
  store i32 %lv , ptr %gep.3, align 4
  %offset.2 = add nsw i64 %iv, %X
  %gep.4 = getelementptr inbounds i32, ptr %ptr.1, i64 %offset.2
  store i32 %lv, ptr %gep.4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %loop.exit, label %loop

loop.exit:
  br label %exit

exit:
  ret void
}

; Alias set with uncomputable bounds contains a single store. We do not need
; runtime checks for that group and it should not block vectorization.
define void @test2_uncomputable_bounds_single_store(ptr noalias %ptr.1, ptr noalias %ptr.2, ptr noalias %ptr.3, i64 %N, i64 %X) {
; CHECK-LABEL: define void @test2_uncomputable_bounds_single_store
; CHECK:       vector.body
; CHECK:         ret void

entry:
  %cond = icmp sgt i64 %N, 0
  br i1 %cond, label %ph, label %exit

ph:
  br label %loop

loop:
  %iv = phi i64 [ 0, %ph ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i32, ptr %ptr.3, i64 %iv
  %offset.1 = load i32, ptr %gep.1, align 4
  %gep.2 = getelementptr inbounds i32, ptr %ptr.2, i32 %offset.1
  store i32 20, ptr %gep.2, align 4
  %gep.3 = getelementptr inbounds i32, ptr %ptr.1, i64 %iv
  store i32 0 , ptr %gep.3, align 4
  %offset.2 = add nsw i64 %iv, %X
  %gep.4 = getelementptr inbounds i32, ptr %ptr.1, i64 %offset.2
  store i32 10, ptr %gep.4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %loop.exit, label %loop

loop.exit:
  br label %exit

exit:
  ret void
}

; Alias set with uncomputable bounds contains a load and a store. This blocks
; vectorization, as we cannot generate runtime-checks for the set.
define void @test3_uncomputable_bounds_load_store(ptr noalias %ptr.1, ptr noalias %ptr.2, ptr noalias %ptr.3, i64 %N, i64 %X) {
; CHECK-LABEL: define void @test3_uncomputable_bounds_load_store
; CHECK-NOT: vector.body

entry:
  %cond = icmp sgt i64 %N, 0
  br i1 %cond, label %ph, label %exit

ph:
  br label %loop

loop:
  %iv = phi i64 [ 0, %ph ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i32, ptr %ptr.3, i64 %iv
  %offset.1 = load i32, ptr %gep.1, align 4
  %gep.2 = getelementptr inbounds i32, ptr %ptr.2, i32 %offset.1
  store i32 20, ptr %gep.2, align 4
  %gep.22 = getelementptr inbounds i32, ptr %ptr.2, i64 %iv
  %lv = load i32, ptr %gep.22, align 4
  %gep.3 = getelementptr inbounds i32, ptr %ptr.1, i64 %iv
  store i32 %lv , ptr %gep.3, align 4
  %offset.2 = add nsw i64 %iv, %X
  %gep.4 = getelementptr inbounds i32, ptr %ptr.1, i64 %offset.2
  store i32 %lv, ptr %gep.4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %loop.exit, label %loop

loop.exit:
  br label %exit

exit:
  ret void
}

; Alias set with uncomputable bounds contains a load and a store. This blocks
; vectorization, as we cannot generate runtime-checks for the set.
define void @test4_uncomputable_bounds_store_store(ptr noalias %ptr.1, ptr noalias %ptr.2, ptr noalias %ptr.3, i64 %N, i64 %X) {
; CHECK-LABEL: define void @test4_uncomputable_bounds_store_store
; CHECK-NOT: vector.body

entry:
  %cond = icmp sgt i64 %N, 0
  br i1 %cond, label %ph, label %exit

ph:
  br label %loop

loop:
  %iv = phi i64 [ 0, %ph ], [ %iv.next, %loop ]
  %gep.1 = getelementptr inbounds i32, ptr %ptr.3, i64 %iv
  %offset.1 = load i32, ptr %gep.1, align 4
  %gep.2 = getelementptr inbounds i32, ptr %ptr.2, i32 %offset.1
  store i32 20, ptr %gep.2, align 4
  %gep.22 = getelementptr inbounds i32, ptr %ptr.2, i64 %iv
  store i32 30, ptr %gep.22, align 4
  %gep.3 = getelementptr inbounds i32, ptr %ptr.1, i64 %iv
  store i32 0 , ptr %gep.3, align 4
  %offset.2 = add nsw i64 %iv, %X
  %gep.4 = getelementptr inbounds i32, ptr %ptr.1, i64 %offset.2
  store i32 10, ptr %gep.4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %loop.exit, label %loop

loop.exit:
  br label %exit

exit:
  ret void
}
