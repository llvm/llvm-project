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
; TODO: We should be able to vectorize this loop once we support vectorizing
; stores of variant values to invariant addresses.
; CHECK-LABEL: inv_val_store_to_inv_address_conditional_diff_values_ic
; CHECK-NOT:   <4 x
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
; TODO: We should vectorize this loop once we relax the check for
; variant/invariant values being stored to invariant address.
; CHECK-LABEL: inv_val_store_to_inv_address_conditional_inv
; CHECK-NOT: <4 x
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

; TODO: This loop can be vectorized once we support variant value being
; stored into invariant address.
; CHECK-LABEL: variant_val_store_to_inv_address
; CHECK-NOT: <4 x i32>
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
  %rdx.lcssa = phi i32 [ %tmp0, %for.body ]
  ret i32 %rdx.lcssa
}
