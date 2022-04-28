; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s

define void @test_chained_first_order_recurrences_1(i16* %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_1
; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_chained_first_order_recurrences_2(i16* %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_2
; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_chained_first_order_recurrences_3(i16* %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3
; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}


define void @test_cyclic_phis(i16* %ptr) {
; CHECK-LABEL: @test_cyclic_phis
; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.2, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_first_order_recurrences_incoming_cycle_preheader(i16* %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_incoming_cycle_preheader
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 0>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, i16* [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, i16* [[TMP1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i16* [[TMP2]] to <4 x i16>*
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, <4 x i16>* [[TMP3]], align 2
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = add <4 x i16> [[TMP4]], <i16 10, i16 10, i16 10, i16 10>
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i16* [[TMP2]] to <4 x i16>*
; CHECK-NEXT:    store <4 x i16> [[TMP5]], <4 x i16>* [[TMP6]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP7]], label %middle.block, label %vector.body
;
entry:
  br label %loop.1

loop.1:
  %p = phi i16 [ 0, %entry ], [ %p, %loop.1 ]
  br i1 true, label %loop, label %loop.1

loop:
  %for.1 = phi i16 [ %p, %loop.1 ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %loop.1 ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add = add i16 %for.1, 10
  store i16 %add, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_chained_first_order_recurrence_sink_users_1(double* %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrence_sink_users_1
; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi double [ 10.0, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi double [ 20.0, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %loop ]
  %add.1 = fadd double 10.0, %for.2
  %add.2 = fadd double %add.1, %for.1
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds double, double* %ptr, i64 %iv
  %for.1.next  = load double, double* %gep.ptr, align 8
  store double %add.2, double* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
