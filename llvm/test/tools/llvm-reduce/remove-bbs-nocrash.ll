; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t

; CHECK-INTERESTINGNESS: ret



define void @f(ptr nocapture %arg, ptr %arg1) {
bb:
  br i1 false, label %bb2, label %bb26

bb2:                                              ; preds = %bb
  br label %bb4

bb4:                                              ; preds = %bb2
  br label %bb5

bb5:                                              ; preds = %bb5, %bb4
  %i = phi i64 [ 0, %bb4 ], [ %i9, %bb5 ]
  %i6 = getelementptr inbounds i64, ptr %arg, i64 %i
  %i7 = load i64, ptr %i6, align 8
  %i8 = getelementptr inbounds i64, ptr %arg1, i64 %i
  store i64 0, ptr %i8, align 8
  %i9 = add nuw nsw i64 %i, 1
  %i10 = icmp eq i64 %i9, 0
  br i1 %i10, label %bb26, label %bb5

bb26:                                             ; preds = %bb5, %bb
  ret void
}
