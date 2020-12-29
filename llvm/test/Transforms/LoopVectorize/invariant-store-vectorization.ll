; RUN: opt < %s -licm -loop-vectorize -force-vector-width=4 -dce -instcombine -licm -S | FileCheck %s

; First licm pass is to hoist/sink invariant stores if possible. Today LICM does
; not hoist/sink the invariant stores. Even if that changes, we should still
; vectorize this loop in case licm is not run.

; The next licm pass after vectorization is to hoist/sink loop invariant
; instructions.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; all tests check that it is legal to vectorize the stores to invariant
; address.


; CHECK-LABEL: inv_val_store_to_inv_address_with_reduction(
; memory check is found.conflict = b[max(n-1,1)] > a && (i8* a)+1 > (i8* b)
; CHECK: vector.memcheck:
; CHECK:    found.conflict

; CHECK-LABEL: vector.body:
; CHECK:         %vec.phi = phi <4 x i32>  [ zeroinitializer, %vector.ph ], [ [[ADD:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         %wide.load = load <4 x i32>
; CHECK:         [[ADD]] = add <4 x i32> %vec.phi, %wide.load
; CHECK-NEXT:    store i32 %ntrunc, i32* %a
; CHECK-NEXT:    %index.next = add i64 %index, 4
; CHECK-NEXT:    icmp eq i64 %index.next, %n.vec
; CHECK-NEXT:    br i1

; CHECK-LABEL: middle.block:
; CHECK:         %rdx.shuf = shufflevector <4 x i32>
define i32 @inv_val_store_to_inv_address_with_reduction(i32* %a, i64 %n, i32* %b) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %tmp3 = add i32 %tmp0, %tmp2
  store i32 %ntrunc, i32* %a
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  ret i32 %tmp4
}

; CHECK-LABEL: inv_val_store_to_inv_address(
; CHECK-LABEL: vector.body:
; CHECK:         store i32 %ntrunc, i32* %a
; CHECK:         store <4 x i32>
; CHECK-NEXT:    %index.next = add i64 %index, 4
; CHECK-NEXT:    icmp eq i64 %index.next, %n.vec
; CHECK-NEXT:    br i1
define void @inv_val_store_to_inv_address(i32* %a, i64 %n, i32* %b) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  store i32 %ntrunc, i32* %a
  store i32 %ntrunc, i32* %tmp1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


; Both of these tests below are handled as predicated stores.

; Conditional store
; if (b[i] == k) a = ntrunc
; TODO: We can be better with the code gen for the first test and we can have
; just one scalar store if vector.or.reduce(vector_cmp(b[i] == k)) is 1.

; CHECK-LABEL:inv_val_store_to_inv_address_conditional(
; CHECK-LABEL: vector.body:
; CHECK:           %wide.load = load <4 x i32>, <4 x i32>*
; CHECK:           [[CMP:%[a-zA-Z0-9.]+]] = icmp eq <4 x i32> %wide.load, %{{.*}}
; CHECK:           store <4 x i32>
; CHECK-NEXT:      [[EE:%[a-zA-Z0-9.]+]] =  extractelement <4 x i1> [[CMP]], i32 0
; CHECK-NEXT:      br i1 [[EE]], label %pred.store.if, label %pred.store.continue

; CHECK-LABEL: pred.store.if:
; CHECK-NEXT:      store i32 %ntrunc, i32* %a
; CHECK-NEXT:      br label %pred.store.continue

; CHECK-LABEL: pred.store.continue:
; CHECK-NEXT:      [[EE1:%[a-zA-Z0-9.]+]] =  extractelement <4 x i1> [[CMP]], i32 1
define void @inv_val_store_to_inv_address_conditional(i32* %a, i64 %n, i32* %b, i32 %k) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %latch ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %cmp = icmp eq i32 %tmp2, %k
  store i32 %ntrunc, i32* %tmp1
  br i1 %cmp, label %cond_store, label %latch

cond_store:
  store i32 %ntrunc, i32* %a
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; if (b[i] == k)
;    a = ntrunc
; else a = k;
; TODO: We could vectorize this once we support multiple uniform stores to the
; same address.
; CHECK-LABEL:inv_val_store_to_inv_address_conditional_diff_values(
; CHECK-NOT:           load <4 x i32>
define void @inv_val_store_to_inv_address_conditional_diff_values(i32* %a, i64 %n, i32* %b, i32 %k) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %latch ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %cmp = icmp eq i32 %tmp2, %k
  store i32 %ntrunc, i32* %tmp1
  br i1 %cmp, label %cond_store, label %cond_store_k

cond_store:
  store i32 %ntrunc, i32* %a
  br label %latch

cond_store_k:
  store i32 %k, i32 * %a
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; Instcombine'd version of above test. Now the store is no longer of invariant
; value.
; scalar store the value extracted from the last element of the vector value.
; CHECK-LABEL: inv_val_store_to_inv_address_conditional_diff_values_ic
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[NTRUNC:%.*]] = trunc i64 [[N:%.*]] to i32
; CHECK-NEXT:    [[TMP0:%.*]] = icmp sgt i64 [[N]], 1
; CHECK-NEXT:    [[SMAX:%.*]] = select i1 [[TMP0]], i64 [[N]], i64 1
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[SMAX]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[A4:%.*]] = bitcast i32* [[A:%.*]] to i8*
; CHECK-NEXT:    [[B1:%.*]] = bitcast i32* [[B:%.*]] to i8*
; CHECK-NEXT:    [[TMP1:%.*]] = icmp sgt i64 [[N]], 1
; CHECK-NEXT:    [[SMAX2:%.*]] = select i1 [[TMP1]], i64 [[N]], i64 1
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i32, i32* [[B]], i64 [[SMAX2]]
; CHECK-NEXT:    [[UGLYGEP:%.*]] = getelementptr i8, i8* [[A4]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ugt i8* [[UGLYGEP]], [[B1]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ugt i32* [[SCEVGEP]], [[A]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[SMAX]], 9223372036854775804
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT5:%.*]] = insertelement <4 x i32> poison, i32 [[K:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT6:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT5]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT7:%.*]] = insertelement <4 x i32> poison, i32 [[NTRUNC]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT8:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT7]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to <4 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, <4 x i32>* [[TMP3]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], [[BROADCAST_SPLAT6]]
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP2]] to <4 x i32>*
; CHECK-NEXT:    store <4 x i32> [[BROADCAST_SPLAT8]], <4 x i32>* [[TMP5]], align 4
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP4]], <4 x i32> [[BROADCAST_SPLAT8]], <4 x i32> [[BROADCAST_SPLAT6]]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x i32> [[PREDPHI]], i32 3
; CHECK-NEXT:    store i32 [[TMP6]], i32* [[A]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP7]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[SMAX]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ], [ 0, [[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ [[I_NEXT:%.*]], [[LATCH:%.*]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[I]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[TMP1]], align 8
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP2]], [[K]]
; CHECK-NEXT:    store i32 [[NTRUNC]], i32* [[TMP1]], align 4
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_STORE:%.*]], label [[COND_STORE_K:%.*]]
; CHECK:       cond_store:
; CHECK-NEXT:    br label [[LATCH]]
; CHECK:       cond_store_k:
; CHECK-NEXT:    br label [[LATCH]]
; CHECK:       latch:
; CHECK-NEXT:    [[STOREVAL:%.*]] = phi i32 [ [[NTRUNC]], [[COND_STORE]] ], [ [[K]], [[COND_STORE_K]] ]
; CHECK-NEXT:    store i32 [[STOREVAL]], i32* [[A]], align 4
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[COND:%.*]] = icmp slt i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[COND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT:%.*]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
define void @inv_val_store_to_inv_address_conditional_diff_values_ic(i32* %a, i64 %n, i32* %b, i32 %k) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %latch ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %cmp = icmp eq i32 %tmp2, %k
  store i32 %ntrunc, i32* %tmp1
  br i1 %cmp, label %cond_store, label %cond_store_k

cond_store:
  br label %latch

cond_store_k:
  br label %latch

latch:
  %storeval = phi i32 [ %ntrunc, %cond_store ], [ %k, %cond_store_k ]
  store i32 %storeval, i32* %a
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; invariant val stored to invariant address predicated on invariant condition
; This is not treated as a predicated store since the block the store belongs to
; is the latch block (which doesn't need to be predicated).
; variant/invariant values being stored to invariant address.
; test checks that the last element of the phi is extracted and scalar stored
; into the uniform address within the loop.
; Since the condition and the phi is loop invariant, they are LICM'ed after
; vectorization.
; CHECK-LABEL: inv_val_store_to_inv_address_conditional_inv
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[NTRUNC:%.*]] = trunc i64 [[N:%.*]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[NTRUNC]], [[K:%.*]]
; CHECK-NEXT:    [[TMP0:%.*]] = icmp sgt i64 [[N]], 1
; CHECK-NEXT:    [[SMAX:%.*]] = select i1 [[TMP0]], i64 [[N]], i64 1
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[SMAX]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[A4:%.*]] = bitcast i32* [[A:%.*]] to i8*
; CHECK-NEXT:    [[B1:%.*]] = bitcast i32* [[B:%.*]] to i8*
; CHECK-NEXT:    [[TMP1:%.*]] = icmp sgt i64 [[N]], 1
; CHECK-NEXT:    [[SMAX2:%.*]] = select i1 [[TMP1]], i64 [[N]], i64 1
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i32, i32* [[B]], i64 [[SMAX2]]
; CHECK-NEXT:    [[UGLYGEP:%.*]] = getelementptr i8, i8* [[A4]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ugt i8* [[UGLYGEP]], [[B1]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ugt i32* [[SCEVGEP]], [[A]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[SMAX]], 9223372036854775804
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT5:%.*]] = insertelement <4 x i32> poison, i32 [[NTRUNC]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT6:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT5]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i1> undef, i1 [[CMP]], i32 3
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x i32> poison, i32 [[K]], i32 3
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP2]], <4 x i32> [[BROADCAST_SPLAT6]], <4 x i32> [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = extractelement <4 x i32> [[PREDPHI]], i32 3
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i32* [[TMP6]] to <4 x i32>*
; CHECK-NEXT:    store <4 x i32> [[BROADCAST_SPLAT6]], <4 x i32>* [[TMP7]], align 4
; CHECK-NEXT:    store i32 [[TMP5]], i32* [[A]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP8]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[SMAX]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ], [ 0, [[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ [[I_NEXT:%.*]], [[LATCH:%.*]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[I]]
; CHECK-NEXT:    store i32 [[NTRUNC]], i32* [[TMP1]], align 4
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_STORE:%.*]], label [[COND_STORE_K:%.*]]
; CHECK:       cond_store:
; CHECK-NEXT:    br label [[LATCH]]
; CHECK:       cond_store_k:
; CHECK-NEXT:    br label [[LATCH]]
; CHECK:       latch:
; CHECK-NEXT:    [[STOREVAL:%.*]] = phi i32 [ [[NTRUNC]], [[COND_STORE]] ], [ [[K]], [[COND_STORE_K]] ]
; CHECK-NEXT:    store i32 [[STOREVAL]], i32* [[A]], align 4
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[COND:%.*]] = icmp slt i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[COND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT:%.*]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
define void @inv_val_store_to_inv_address_conditional_inv(i32* %a, i64 %n, i32* %b, i32 %k) {
entry:
  %ntrunc = trunc i64 %n to i32
  %cmp = icmp eq i32 %ntrunc, %k
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %latch ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  store i32 %ntrunc, i32* %tmp1
  br i1 %cmp, label %cond_store, label %cond_store_k

cond_store:
  br label %latch

cond_store_k:
  br label %latch

latch:
  %storeval = phi i32 [ %ntrunc, %cond_store ], [ %k, %cond_store_k ]
  store i32 %storeval, i32* %a
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; variant value stored to uniform address tests that the code gen extracts the
; last element from the variant vector and scalar stores it into the uniform
; address.
; CHECK-LABEL: variant_val_store_to_inv_address
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = icmp sgt i64 [[N:%.*]], 1
; CHECK-NEXT:    [[SMAX:%.*]] = select i1 [[TMP0]], i64 [[N]], i64 1
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[SMAX]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[B2:%.*]] = bitcast i32* [[B:%.*]] to i8*
; CHECK-NEXT:    [[A1:%.*]] = bitcast i32* [[A:%.*]] to i8*
; CHECK-NEXT:    [[UGLYGEP:%.*]] = getelementptr i8, i8* [[A1]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = icmp sgt i64 [[N]], 1
; CHECK-NEXT:    [[SMAX3:%.*]] = select i1 [[TMP1]], i64 [[N]], i64 1
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i32, i32* [[B]], i64 [[SMAX3]]
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ugt i32* [[SCEVGEP]], [[A]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ugt i8* [[UGLYGEP]], [[B2]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[SMAX]], 9223372036854775804
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to <4 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, <4 x i32>* [[TMP3]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <4 x i32> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    store i32 [[TMP4]], i32* [[A]], align 4
; CHECK-NEXT:    [[TMP5]] = add <4 x i32> [[VEC_PHI]], [[WIDE_LOAD]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[DOTLCSSA:%.*]] = phi <4 x i32> [ [[TMP5]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[DOTLCSSA]], <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <4 x i32> [[DOTLCSSA]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF5:%.*]] = shufflevector <4 x i32> [[BIN_RDX]], <4 x i32> poison, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX6:%.*]] = add <4 x i32> [[BIN_RDX]], [[RDX_SHUF5]]
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <4 x i32> [[BIN_RDX6]], i32 0
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[SMAX]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ], [ 0, [[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ [[TMP7]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY]] ], [ 0, [[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ [[I_NEXT:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = phi i32 [ [[TMP3:%.*]], [[FOR_BODY]] ], [ [[BC_MERGE_RDX]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[I]]
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[TMP1]], align 8
; CHECK-NEXT:    store i32 [[TMP2]], i32* [[A]], align 4
; CHECK-NEXT:    [[TMP3]] = add i32 [[TMP0]], [[TMP2]]
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[COND:%.*]] = icmp slt i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[COND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT:%.*]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    [[TMP3_LCSSA:%.*]] = phi i32 [ [[TMP3]], [[FOR_BODY]] ]
; CHECK-NEXT:    br label [[FOR_END]]
define i32 @variant_val_store_to_inv_address(i32* %a, i64 %n, i32* %b, i32 %k) {
entry:
  %ntrunc = trunc i64 %n to i32
  %cmp = icmp eq i32 %ntrunc, %k
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  store i32 %tmp2, i32* %a
  %tmp3 = add i32 %tmp0, %tmp2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %rdx.lcssa = phi i32 [ %tmp3, %for.body ]
  ret i32 %rdx.lcssa
}

; Multiple variant stores to the same uniform address
; We do not vectorize such loops currently.
;  for(; i < itr; i++) {
;    for(; j < itr; j++) {
;      var1[i] = var2[j] + var1[i];
;      var1[i]++;
;    }
;  }

; CHECK-LABEL: multiple_uniform_stores
; CHECK-NOT:     <4 x i32>
define i32 @multiple_uniform_stores(i32* nocapture %var1, i32* nocapture readonly %var2, i32 %itr) #0 {
entry:
  %cmp20 = icmp eq i32 %itr, 0
  br i1 %cmp20, label %for.end10, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %for.inc8 ], [ 0, %entry ]
  %j.022 = phi i32 [ %j.1.lcssa, %for.inc8 ], [ 0, %entry ]
  %cmp218 = icmp ult i32 %j.022, %itr
  br i1 %cmp218, label %for.body3.lr.ph, label %for.inc8

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %arrayidx5 = getelementptr inbounds i32, i32* %var1, i64 %indvars.iv23
  %0 = zext i32 %j.022 to i64
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ %0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds i32, i32* %var2, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %2 = load i32, i32* %arrayidx5, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, i32* %arrayidx5, align 4
  %3 = load i32, i32* %arrayidx5, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.022, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv.next24 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %itr
  br i1 %exitcond26, label %for.end10, label %for.cond1.preheader

for.end10:                                        ; preds = %for.inc8, %entry
  ret i32 undef
}

; second uniform store to the same address is conditional.
; we do not vectorize this.
; CHECK-LABEL: multiple_uniform_stores_conditional
; CHECK-NOT:    <4 x i32>
define i32 @multiple_uniform_stores_conditional(i32* nocapture %var1, i32* nocapture readonly %var2, i32 %itr) #0 {
entry:
  %cmp20 = icmp eq i32 %itr, 0
  br i1 %cmp20, label %for.end10, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %for.inc8 ], [ 0, %entry ]
  %j.022 = phi i32 [ %j.1.lcssa, %for.inc8 ], [ 0, %entry ]
  %cmp218 = icmp ult i32 %j.022, %itr
  br i1 %cmp218, label %for.body3.lr.ph, label %for.inc8

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %arrayidx5 = getelementptr inbounds i32, i32* %var1, i64 %indvars.iv23
  %0 = zext i32 %j.022 to i64
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ %0, %for.body3.lr.ph ], [ %indvars.iv.next, %latch ]
  %arrayidx = getelementptr inbounds i32, i32* %var2, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %2 = load i32, i32* %arrayidx5, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, i32* %arrayidx5, align 4
  %3 = load i32, i32* %arrayidx5, align 4
  %4 = add nsw i32 %3, 1
  %5 = icmp ugt i32 %3, 42
  br i1 %5, label %cond_store, label %latch

cond_store:
  store i32 %4, i32* %arrayidx5, align 4
  br label %latch

latch:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.022, %for.cond1.preheader ], [ %itr, %latch ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv.next24 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %itr
  br i1 %exitcond26, label %for.end10, label %for.cond1.preheader

for.end10:                                        ; preds = %for.inc8, %entry
  ret i32 undef
}

; cannot vectorize loop with unsafe dependency between uniform load (%tmp10) and store
; (%tmp12) to the same address
; PR39653
; Note: %tmp10 could be replaced by phi(%arg4, %tmp12), a potentially vectorizable
; 1st-order-recurrence
define void @unsafe_dep_uniform_load_store(i32 %arg, i32 %arg1, i64 %arg2, i16* %arg3, i32 %arg4, i64 %arg5) {
; CHECK-LABEL: unsafe_dep_uniform_load_store
; CHECK-NOT: <4 x i32>
bb:
  %tmp = alloca i32
  store i32 %arg4, i32* %tmp
  %tmp6 = getelementptr inbounds i16, i16* %arg3, i64 %arg5
  br label %bb7

bb7:
  %tmp8 = phi i64 [ 0, %bb ], [ %tmp24, %bb7 ]
  %tmp9 = phi i32 [ %arg1, %bb ], [ %tmp23, %bb7 ]
  %tmp10 = load i32, i32* %tmp
  %tmp11 = mul nsw i32 %tmp9, %tmp10
  %tmp12 = srem i32 %tmp11, 65536
  %tmp13 = add nsw i32 %tmp12, %tmp9
  %tmp14 = trunc i32 %tmp13 to i16
  %tmp15 = trunc i64 %tmp8 to i32
  %tmp16 = add i32 %arg, %tmp15
  %tmp17 = zext i32 %tmp16 to i64
  %tmp18 = getelementptr inbounds i16, i16* %tmp6, i64 %tmp17
  store i16 %tmp14, i16* %tmp18, align 2
  %tmp19 = add i32 %tmp13, %tmp9
  %tmp20 = trunc i32 %tmp19 to i16
  %tmp21 = and i16 %tmp20, 255
  %tmp22 = getelementptr inbounds i16, i16* %arg3, i64 %tmp17
  store i16 %tmp21, i16* %tmp22, align 2
  %tmp23 = add nsw i32 %tmp9, 1
  %tmp24 = add nuw nsw i64 %tmp8, 1
  %tmp25 = icmp eq i64 %tmp24, %arg2
  store i32 %tmp12, i32* %tmp
  br i1 %tmp25, label %bb26, label %bb7

bb26:
  ret void
}
