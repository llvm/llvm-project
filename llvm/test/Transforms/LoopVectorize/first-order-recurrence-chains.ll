; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s

define i16 @test_chained_first_order_recurrences_1(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_1
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = add <4 x i16> [[TMP4]], [[TMP5]]
; CHECK-NEXT:    store <4 x i16> [[TMP6]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP8]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT2:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = add i16 %for.1, %for.2
  ret i16 %res
}

define i16 @test_chained_first_order_recurrences_2(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_2
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = add <4 x i16> [[TMP4]], [[TMP5]]
; CHECK-NEXT:    store <4 x i16> [[TMP6]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP8]], label %middle.block, label %vector.body, !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT2:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI3:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
;
entry:
  br label %loop

loop:
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = add i16 %for.1, %for.2
  ret i16 %res
}

define i16 @test_chained_first_order_recurrences_3(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP5:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR2]], <4 x i16> [[TMP5]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP7:%.*]] = add <4 x i16> [[TMP4]], [[TMP5]]
; CHECK-NEXT:    [[TMP8:%.*]] = add <4 x i16> [[TMP7]], [[TMP6]]
; CHECK-NEXT:    store <4 x i16> [[TMP8]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI4:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT7:%.*]] = extractelement <4 x i16> [[TMP5]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI8:%.*]] = extractelement <4 x i16> [[TMP5]], i32 2
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res.1 = add i16 %for.1, %for.2
  %res.2 = add i16 %res.1, %for.3
  ret i16 %res.2
}

define void @test_cyclic_phis(ptr %ptr) {
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
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_first_order_recurrences_incoming_cycle_preheader(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_incoming_cycle_preheader
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 0>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = add <4 x i16> [[TMP4]], <i16 10, i16 10, i16 10, i16 10>
; CHECK-NEXT:    store <4 x i16> [[TMP5]], ptr [[TMP2]], align 2
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
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, 10
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define i16 @test_chained_first_order_recurrences_3_reordered_1(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3_reordered_1
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP5:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR2]], <4 x i16> [[TMP5]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP7:%.*]] = add <4 x i16> [[TMP4]], [[TMP5]]
; CHECK-NEXT:    [[TMP8:%.*]] = add <4 x i16> [[TMP7]], [[TMP6]]
; CHECK-NEXT:    store <4 x i16> [[TMP8]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT7:%.*]] = extractelement <4 x i16> [[TMP5]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI8:%.*]] = extractelement <4 x i16> [[TMP5]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI4:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
;
entry:
  br label %loop

loop:
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res.1 = add i16 %for.1, %for.2
  %res.2 = add i16 %res.1, %for.3
  ret i16 %res.2
}

define i16 @test_chained_first_order_recurrences_3_reordered_2(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3_reordered_2
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP5:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR2]], <4 x i16> [[TMP5]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP7:%.*]] = add <4 x i16> [[TMP4]], [[TMP5]]
; CHECK-NEXT:    [[TMP8:%.*]] = add <4 x i16> [[TMP7]], [[TMP6]]
; CHECK-NEXT:    store <4 x i16> [[TMP8]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI4:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT7:%.*]] = extractelement <4 x i16> [[TMP5]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI8:%.*]] = extractelement <4 x i16> [[TMP5]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
;
entry:
  br label %loop

loop:
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res.1 = add i16 %for.1, %for.2
  %res.2 = add i16 %res.1, %for.3
  ret i16 %res.2
}

define i16 @test_chained_first_order_recurrences_3_for2_no_other_uses(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3_for2_no_other_uses
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP5:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR2]], <4 x i16> [[TMP5]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP7:%.*]] = add <4 x i16> [[TMP4]], <i16 10, i16 10, i16 10, i16 10>
; CHECK-NEXT:    [[TMP8:%.*]] = add <4 x i16> [[TMP7]], [[TMP6]]
; CHECK-NEXT:    store <4 x i16> [[TMP8]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI4:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT7:%.*]] = extractelement <4 x i16> [[TMP5]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI8:%.*]] = extractelement <4 x i16> [[TMP5]], i32 2
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, 10
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res.1 = add i16 %for.1, %for.2
  %res.2 = add i16 %res.1, %for.3
  ret i16 %res.2
}

define i16 @test_chained_first_order_recurrences_3_for1_for2_no_other_uses(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrences_3_for1_for2_no_other_uses
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 22>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 33>, %vector.ph ], [ [[TMP5:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i16, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i16, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x i16>, ptr [[TMP2]], align 2
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x i16> [[VECTOR_RECUR]], <4 x i16> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5]] = shufflevector <4 x i16> [[VECTOR_RECUR1]], <4 x i16> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x i16> [[VECTOR_RECUR2]], <4 x i16> [[TMP5]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP8:%.*]] = add <4 x i16> [[TMP6]], <i16 10, i16 10, i16 10, i16 10>
; CHECK-NEXT:    store <4 x i16> [[TMP8]], ptr [[TMP2]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP10]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:      middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i16> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT3:%.*]] = extractelement <4 x i16> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI4:%.*]] = extractelement <4 x i16> [[TMP4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT7:%.*]] = extractelement <4 x i16> [[TMP5]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI8:%.*]] = extractelement <4 x i16> [[TMP5]], i32 2
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.3, 10
  store i16 %add.1, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res.1 = add i16 %for.1, %for.2
  %res.2 = add i16 %res.1, %for.3
  ret i16 %res.2
}

define double @test_chained_first_order_recurrence_sink_users_1(ptr %ptr) {
; CHECK-LABEL: @test_chained_first_order_recurrence_sink_users_1
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 1.000000e+01>, %vector.ph ], [ [[WIDE_LOAD:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 2.000000e+01>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i64 1, [[INDEX]]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[OFFSET_IDX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds double, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds double, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD]] = load <4 x double>, ptr [[TMP2]], align 8
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x double> [[VECTOR_RECUR]], <4 x double> [[WIDE_LOAD]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <4 x double> [[VECTOR_RECUR1]], <4 x double> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = fadd <4 x double> <double 1.000000e+01, double 1.000000e+01, double 1.000000e+01, double 1.000000e+01>, [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = fadd <4 x double> [[TMP6]], [[TMP4]]
; CHECK-NEXT:    store <4 x double> [[TMP7]], ptr [[TMP2]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP9:%.*]] = icmp eq i64 [[INDEX_NEXT]], 996
; CHECK-NEXT:    br i1 [[TMP9]], label %middle.block, label %vector.body, !llvm.loop [[LOOP10:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 999, 996
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x double> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x double> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT2:%.*]] = extractelement <4 x double> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI3:%.*]] = extractelement <4 x double> [[TMP4]], i32 2
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
  %gep.ptr = getelementptr inbounds double, ptr %ptr, i64 %iv
  %for.1.next  = load double, ptr %gep.ptr, align 8
  store double %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = fadd double %for.1, %for.2
  ret double %res
}

define void @test_first_order_recurrences_and_reduction(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_and_reduction(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %red, %loop ]
  %red = phi i16 [ 33, %entry ], [ %red.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %lv = load i16, ptr %gep.ptr
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, 10
  %red.next = add i16 %red, %lv
  store i16 %add.1, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define i64 @test_first_order_recurrences_and_induction(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_and_induction(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i64> [ <i64 poison, i64 poison, i64 poison, i64 22>, %vector.ph ], [ [[VEC_IND:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <4 x i64> [[VECTOR_RECUR]], <4 x i64> [[VEC_IND]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i64, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = add <4 x i64> [[TMP1]], <i64 10, i64 10, i64 10, i64 10>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i64, ptr [[TMP2]], i32 0
; CHECK-NEXT:    store <4 x i64> [[TMP4]], ptr [[TMP3]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP5]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i64> [[VEC_IND]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i64> [[VEC_IND]], i32 2
; CHECK-NEXT:    br i1 [[CMP_N]]

entry:
  br label %loop

loop:
  %for.1 = phi i64 [ 22, %entry ], [ %iv, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i64, ptr %ptr, i64 %iv
  %add.1 = add i64 %for.1, 10
  store i64 %add.1, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i64 %for.1
}

; Same as @test_first_order_recurrences_and_induction but with order of phis
; flipped.
define i64 @test_first_order_recurrences_and_induction2(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_and_induction2(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i64> [ <i64 poison, i64 poison, i64 poison, i64 22>, %vector.ph ], [ [[VEC_IND]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <4 x i64> [[VECTOR_RECUR]], <4 x i64> [[VEC_IND]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i64, ptr [[PTR:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = add <4 x i64> [[TMP1]], <i64 10, i64 10, i64 10, i64 10>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i64, ptr [[TMP2]], i32 0
; CHECK-NEXT:    store <4 x i64> [[TMP4]], ptr [[TMP3]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP5]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x i64> [[VEC_IND]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x i64> [[VEC_IND]], i32 2
; CHECK-NEXT:    br i1 [[CMP_N]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %for.1 = phi i64 [ 22, %entry ], [ %iv, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i64, ptr %ptr, i64 %iv
  %add.1 = add i64 %for.1, 10
  store i64 %add.1, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i64 %for.1
}

define ptr @test_first_order_recurrences_and_pointer_induction1(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_and_pointer_induction1(
; CHECK:       vector.ph:
; CHECK-NEXT:    [[IND_END:%.*]] = getelementptr i8, ptr [[PTR:%.*]], i64 4000
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[POINTER_PHI:%.*]] = phi ptr [ [[PTR]], %vector.ph ], [ [[PTR_IND:%.*]], %vector.body ]
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x ptr> [ <ptr poison, ptr poison, ptr poison, ptr null>, %vector.ph ], [ [[TMP0:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0]] = getelementptr i8, ptr [[POINTER_PHI]], <4 x i64> <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x ptr> [[VECTOR_RECUR]], <4 x ptr> [[TMP0]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds ptr, ptr [[PTR]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds ptr, ptr [[TMP3]], i32 0
; CHECK-NEXT:    store <4 x ptr> [[TMP0]], ptr [[TMP4]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[PTR_IND]] = getelementptr i8, ptr [[POINTER_PHI]], i64 16
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP5]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x ptr> [[TMP0]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x ptr> [[TMP0]], i32 2
; CHECK-NEXT:    br i1 [[CMP_N]],
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %for.1 = phi ptr [ null, %entry ], [ %ptr.iv, %loop ]
  %ptr.iv = phi ptr [ %ptr, %entry ], [ %ptr.iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds ptr, ptr %ptr, i64 %iv
  store ptr %ptr.iv, ptr %gep.ptr
  %ptr.iv.next = getelementptr i32, ptr %ptr.iv, i64 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret ptr %for.1
}

; same as @test_first_order_recurrences_and_pointer_induction1 but with order
; of phis flipped.
define ptr @test_first_order_recurrences_and_pointer_induction2(ptr %ptr) {
; CHECK-LABEL: @test_first_order_recurrences_and_pointer_induction2(
; CHECK:       vector.ph:
; CHECK-NEXT:    [[IND_END:%.*]] = getelementptr i8, ptr [[PTR:%.*]], i64 4000
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[POINTER_PHI:%.*]] = phi ptr [ [[PTR]], %vector.ph ], [ [[PTR_IND:%.*]], %vector.body ]
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x ptr> [ <ptr poison, ptr poison, ptr poison, ptr null>, %vector.ph ], [ [[TMP0:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0]] = getelementptr i8, ptr [[POINTER_PHI]], <4 x i64> <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x ptr> [[VECTOR_RECUR]], <4 x ptr> [[TMP0]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds ptr, ptr [[PTR]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds ptr, ptr [[TMP3]], i32 0
; CHECK-NEXT:    store <4 x ptr> [[TMP0]], ptr [[TMP4]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[PTR_IND]] = getelementptr i8, ptr [[POINTER_PHI]], i64 16
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP5]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x ptr> [[TMP0]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x ptr> [[TMP0]], i32 2
; CHECK-NEXT:    br i1 [[CMP_N]],
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr.iv = phi ptr [ %ptr, %entry ], [ %ptr.iv.next, %loop ]
  %for.1 = phi ptr [ null, %entry ], [ %ptr.iv, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds ptr, ptr %ptr, i64 %iv
  store ptr %ptr.iv, ptr %gep.ptr
  %ptr.iv.next = getelementptr i32, ptr %ptr.iv, i64 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret ptr %for.1
}

; In this test case, %USE_2_FORS uses 2 different fixed-order recurrences and
; it needs to be sunk past the previous value for both recurrences.
define double @test_resinking_required(ptr %p, ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: @test_resinking_required(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 0.000000e+00>, %vector.ph ], [ [[BROADCAST_SPLAT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR1:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 0.000000e+00>, %vector.ph ], [ [[BROADCAST_SPLAT4:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR2:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 0.000000e+00>, %vector.ph ], [ [[TMP4:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = load double, ptr %a, align 8
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x double> poison, double [[TMP0]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <4 x double> [[VECTOR_RECUR]], <4 x double> [[BROADCAST_SPLAT]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP2:%.*]] = fdiv <4 x double> zeroinitializer, [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = load double, ptr %b, align 8
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <4 x double> poison, double [[TMP3]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT3]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4]] = shufflevector <4 x double> [[VECTOR_RECUR1]], <4 x double> [[BROADCAST_SPLAT4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <4 x double> [[VECTOR_RECUR2]], <4 x double> [[TMP4]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x double> [[TMP2]], i32 3
; CHECK-NEXT:    store double [[TMP6]], ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], 0
; CHECK-NEXT:    br i1 [[TMP7]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP28:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 0, 0
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT:%.*]] = extractelement <4 x double> [[BROADCAST_SPLAT]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI:%.*]] = extractelement <4 x double> [[BROADCAST_SPLAT]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT5:%.*]] = extractelement <4 x double> [[BROADCAST_SPLAT4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI6:%.*]] = extractelement <4 x double> [[BROADCAST_SPLAT4]], i32 2
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT9:%.*]] = extractelement <4 x double> [[TMP4]], i32 3
; CHECK-NEXT:    [[VECTOR_RECUR_EXTRACT_FOR_PHI10:%.*]] = extractelement <4 x double> [[TMP4]], i32 2
; CHECK-NEXT:    br i1 [[CMP_N]], label %End, label %scalar.ph
;
Entry:
  br label %Loop

Loop:
  %for.1 = phi double [ %l1, %Loop ], [ 0.000000e+00, %Entry ]
  %for.2 = phi double [ %l2, %Loop ], [ 0.000000e+00, %Entry ]
  %for.3 = phi double [ %for.2, %Loop ], [ 0.000000e+00, %Entry ]
  %iv = phi i64 [ %iv.next, %Loop ], [ 0, %Entry ]
  %USE_2_FORS = fdiv double %for.3, %for.1
  %div = fdiv double 0.000000e+00, %for.1
  %l1 = load double, ptr %a, align 8
  %iv.next= add nuw nsw i64 %iv, 1
  %l2 = load double, ptr %b, align 8
  store double %div, ptr %p, align 8
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %End, label %Loop

End:
  %res.1 = fadd double %for.1, %for.2
  %res.2 = fadd double %res.1, %for.3
  ret double %res.2
}
