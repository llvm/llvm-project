; RUN: opt -S -passes=gvn < %s | FileCheck %s
; RUN: opt -S -passes='gvn<memoryssa>' < %s | FileCheck %s

; Reduced from a std::min_element over an index vector with a comparator that
; loads data[a] < data[b]. The best index is loop-carried through a select, and
; the load of data[best_index] should be carried in lockstep.
define float @min_element_indexed(ptr nocapture readonly %indices,
                                  ptr nocapture readonly %data,
                                  ptr readonly %last) {
; CHECK-LABEL: define float @min_element_indexed(
; CHECK:       entry:
; CHECK:         [[BEST_IDX_PRE:%.*]] = load i32, ptr [[INDICES:%.*]], align 4
; CHECK:         [[BEST_IDX_ZEXT_PRE:%.*]] = zext i32 [[BEST_IDX_PRE]] to i64
; CHECK:         [[BEST_DATA_PTR_PRE:%.*]] = getelementptr inbounds float, ptr [[DATA:%.*]], i64 [[BEST_IDX_ZEXT_PRE]]
; CHECK:         [[BESTVAL_PRE:%.*]] = load float, ptr [[BEST_DATA_PTR_PRE]], align 4
; CHECK:       loop:
; CHECK:         [[BESTVAL_PHI:%.*]] = phi float [ [[BESTVAL_PRE]], %entry ], [ [[BESTVAL_NEXT:%.*]], %loop ]
; CHECK:         [[PVAL:%.*]] = load float, ptr [[P_DATA_PTR:%.*]], align 4
; CHECK-NOT:     load float
; CHECK:         [[CMP:%.*]] = fcmp olt float [[PVAL]], [[BESTVAL_PHI]]
; CHECK:         [[BESTVAL_NEXT]] = select i1 [[CMP]], float [[PVAL]], float [[BESTVAL_PHI]]
; CHECK:       exit:
; CHECK:         ret float [[BESTVAL_PHI]]
entry:
  %p0 = getelementptr inbounds i32, ptr %indices, i64 1
  br label %loop

loop:
  %p = phi ptr [ %p0, %entry ], [ %inc, %loop ]
  %best = phi ptr [ %indices, %entry ], [ %best.next, %loop ]
  %idx = load i32, ptr %p, align 4
  %idx.zext = zext i32 %idx to i64
  %p.data.ptr = getelementptr inbounds float, ptr %data, i64 %idx.zext
  %pval = load float, ptr %p.data.ptr, align 4
  %best.idx = load i32, ptr %best, align 4
  %best.idx.zext = zext i32 %best.idx to i64
  %best.data.ptr = getelementptr inbounds float, ptr %data, i64 %best.idx.zext
  %bestval = load float, ptr %best.data.ptr, align 4
  %cmp = fcmp olt float %pval, %bestval
  %best.next = select i1 %cmp, ptr %p, ptr %best
  %inc = getelementptr inbounds i32, ptr %p, i64 1
  %done = icmp eq ptr %inc, %last
  br i1 %done, label %exit, label %loop

exit:
  ret float %bestval
}

define float @min_element_indexed_loop_store(ptr nocapture readonly %indices,
                                             ptr nocapture %data,
                                             ptr %last,
                                             ptr %sink) {
; CHECK-LABEL: define float @min_element_indexed_loop_store(
; CHECK:       entry:
; CHECK-NOT:     load float
; CHECK:         br label %loop
; CHECK:       loop:
; CHECK:         store i32 0, ptr [[SINK:%.*]], align 4
; CHECK:         [[BESTVAL:%.*]] = load float, ptr [[BEST_DATA_PTR:%.*]], align 4
; CHECK:       exit:
; CHECK:         ret float [[BESTVAL]]
entry:
  %p0 = getelementptr inbounds i32, ptr %indices, i64 1
  br label %loop

loop:
  %p = phi ptr [ %p0, %entry ], [ %inc, %loop ]
  %best = phi ptr [ %indices, %entry ], [ %best.next, %loop ]
  %idx = load i32, ptr %p, align 4
  %idx.zext = zext i32 %idx to i64
  %p.data.ptr = getelementptr inbounds float, ptr %data, i64 %idx.zext
  %pval = load float, ptr %p.data.ptr, align 4
  store i32 0, ptr %sink, align 4
  %best.idx = load i32, ptr %best, align 4
  %best.idx.zext = zext i32 %best.idx to i64
  %best.data.ptr = getelementptr inbounds float, ptr %data, i64 %best.idx.zext
  %bestval = load float, ptr %best.data.ptr, align 4
  %cmp = fcmp olt float %pval, %bestval
  %best.next = select i1 %cmp, ptr %p, ptr %best
  %inc = getelementptr inbounds i32, ptr %p, i64 1
  %done = icmp eq ptr %inc, %last
  br i1 %done, label %exit, label %loop

exit:
  ret float %bestval
}

define float @min_element_indexed_nneg_cast(ptr nocapture readonly %indices,
                                            ptr nocapture readonly %data,
                                            ptr readonly %last) {
; CHECK-LABEL: define float @min_element_indexed_nneg_cast(
; CHECK:       loop:
; CHECK:         [[BEST_IDX:%.*]] = phi i32
; CHECK:         [[PVAL:%.*]] = load float, ptr [[P_DATA_PTR:%.*]], align 4
; CHECK:         [[BEST_IDX_ZEXT:%.*]] = zext nneg i32 [[BEST_IDX]] to i64
; CHECK:         [[BEST_DATA_PTR:%.*]] = getelementptr inbounds float, ptr [[DATA:%.*]], i64 [[BEST_IDX_ZEXT]]
; CHECK:         [[BESTVAL:%.*]] = load float, ptr [[BEST_DATA_PTR]], align 4
; CHECK:       exit:
; CHECK:         ret float [[BESTVAL]]
entry:
  %p0 = getelementptr inbounds i32, ptr %indices, i64 1
  br label %loop

loop:
  %p = phi ptr [ %p0, %entry ], [ %inc, %loop ]
  %best = phi ptr [ %indices, %entry ], [ %best.next, %loop ]
  %idx = load i32, ptr %p, align 4
  %idx.zext = zext nneg i32 %idx to i64
  %p.data.ptr = getelementptr inbounds float, ptr %data, i64 %idx.zext
  %pval = load float, ptr %p.data.ptr, align 4
  %best.idx = load i32, ptr %best, align 4
  %best.idx.zext = zext nneg i32 %best.idx to i64
  %best.data.ptr = getelementptr inbounds float, ptr %data, i64 %best.idx.zext
  %bestval = load float, ptr %best.data.ptr, align 4
  %cmp = fcmp olt float %pval, %bestval
  %best.next = select i1 %cmp, ptr %p, ptr %best
  %inc = getelementptr inbounds i32, ptr %p, i64 1
  %done = icmp eq ptr %inc, %last
  br i1 %done, label %exit, label %loop

exit:
  ret float %bestval
}
