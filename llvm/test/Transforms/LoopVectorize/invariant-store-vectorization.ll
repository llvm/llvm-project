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
; CHECK-NEXT:    %index.next = add nuw i64 %index, 4
; CHECK-NEXT:    icmp eq i64 %index.next, %n.vec
; CHECK-NEXT:    br i1

; CHECK-LABEL: middle.block:
; CHECK:         call i32 @llvm.vector.reduce.add.v4i32(<4 x i32>
define i32 @inv_val_store_to_inv_address_with_reduction(i32* %a, i64 %n, i32* %b) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %i0 = phi i32 [ %i3, %for.body ], [ 0, %entry ]
  %i1 = getelementptr inbounds i32, i32* %b, i64 %i
  %i2 = load i32, i32* %i1, align 8
  %i3 = add i32 %i0, %i2
  store i32 %ntrunc, i32* %a
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %i4 = phi i32 [ %i3, %for.body ]
  ret i32 %i4
}

; CHECK-LABEL: inv_val_store_to_inv_address(
; CHECK-LABEL: vector.body:
; CHECK:         store i32 %ntrunc, i32* %a
; CHECK:         store <4 x i32>
; CHECK-NEXT:    %index.next = add nuw i64 %index, 4
; CHECK-NEXT:    icmp eq i64 %index.next, %n.vec
; CHECK-NEXT:    br i1
define void @inv_val_store_to_inv_address(i32* %a, i64 %n, i32* %b) {
entry:
  %ntrunc = trunc i64 %n to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %i1 = getelementptr inbounds i32, i32* %b, i64 %i
  %i2 = load i32, i32* %i1, align 8
  store i32 %ntrunc, i32* %a
  store i32 %ntrunc, i32* %i1
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
  %i1 = getelementptr inbounds i32, i32* %b, i64 %i
  %i2 = load i32, i32* %i1, align 8
  %cmp = icmp eq i32 %i2, %k
  store i32 %ntrunc, i32* %i1
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
  %i1 = getelementptr inbounds i32, i32* %b, i64 %i
  %i2 = load i32, i32* %i1, align 8
  %cmp = icmp eq i32 %i2, %k
  store i32 %ntrunc, i32* %i1
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

; cannot vectorize loop with unsafe dependency between uniform load (%i10) and store
; (%i12) to the same address
; PR39653
; Note: %i10 could be replaced by phi(%arg4, %i12), a potentially vectorizable
; 1st-order-recurrence
define void @unsafe_dep_uniform_load_store(i32 %arg, i32 %arg1, i64 %arg2, i16* %arg3, i32 %arg4, i64 %arg5) {
; CHECK-LABEL: unsafe_dep_uniform_load_store
; CHECK-NOT: <4 x i32>
bb:
  %i = alloca i32
  store i32 %arg4, i32* %i
  %i6 = getelementptr inbounds i16, i16* %arg3, i64 %arg5
  br label %bb7

bb7:
  %i8 = phi i64 [ 0, %bb ], [ %i24, %bb7 ]
  %i9 = phi i32 [ %arg1, %bb ], [ %i23, %bb7 ]
  %i10 = load i32, i32* %i
  %i11 = mul nsw i32 %i9, %i10
  %i12 = srem i32 %i11, 65536
  %i13 = add nsw i32 %i12, %i9
  %i14 = trunc i32 %i13 to i16
  %i15 = trunc i64 %i8 to i32
  %i16 = add i32 %arg, %i15
  %i17 = zext i32 %i16 to i64
  %i18 = getelementptr inbounds i16, i16* %i6, i64 %i17
  store i16 %i14, i16* %i18, align 2
  %i19 = add i32 %i13, %i9
  %i20 = trunc i32 %i19 to i16
  %i21 = and i16 %i20, 255
  %i22 = getelementptr inbounds i16, i16* %arg3, i64 %i17
  store i16 %i21, i16* %i22, align 2
  %i23 = add nsw i32 %i9, 1
  %i24 = add nuw nsw i64 %i8, 1
  %i25 = icmp eq i64 %i24, %arg2
  store i32 %i12, i32* %i
  br i1 %i25, label %bb26, label %bb7

bb26:
  ret void
}

; Make sure any check-not directives are not triggered by function declarations.
; CHECK: declare
