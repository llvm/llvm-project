; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s --check-prefixes=COMMON,DEFAULT
; RUN: opt < %s -passes=loop-vectorize -tail-predication=enabled -prefer-predicate-over-epilogue=predicate-dont-vectorize -S | FileCheck %s --check-prefixes=COMMON,CHECK-TF,CHECK-PREFER
; RUN: opt < %s -passes=loop-vectorize -tail-predication=enabled -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -S | FileCheck %s --check-prefixes=COMMON,CHECK-TF,CHECK-PREFER
; RUN: opt < %s -passes=loop-vectorize -tail-predication=enabled -S | FileCheck %s --check-prefixes=COMMON,CHECK-TF,CHECK-ENABLE-TP

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-unknown-eabihf"

; This IR corresponds to this type of C-code:
;
;  void f(char *a, char *b, char *c, int N) {
;    while (N-- > 0)
;      *c++ = *a++ + *b++;
;  }
;
define dso_local void @sgt_loopguard(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_loopguard(
; COMMON:       vector.body:

; CHECK-TF:     %[[VIVELEM0:.*]] = extractelement <16 x i32> %vec.iv, i32 0
; CHECK-TF:     %active.lane.mask = call <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32 %[[VIVELEM0]], i32 %N)
; CHECK-TF:     llvm.masked.load.v16i8.p0(ptr %{{.*}}, i32 1, <16 x i1> %active.lane.mask
; CHECK-TF:     llvm.masked.load.v16i8.p0(ptr %{{.*}}, i32 1, <16 x i1> %active.lane.mask
; CHECK-TF:     llvm.masked.store.v16i8.p0(<16 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %active.lane.mask)
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %while.body.preheader, label %while.end

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %while.body.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %while.body.preheader ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %while.body.preheader ]
  %dec = add nsw i32 %N.addr.09, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

; No loop-guard: we need one for this to be valid.
;
define dso_local void @sgt_no_loopguard(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_no_loopguard(
; COMMON:       vector.body:
; CHECK-TF:     masked.load
; CHECK-TF:     masked.load
; CHECK-TF:     masked.store
entry:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ %N, %entry ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %entry ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %entry ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %entry ]
  %dec = add nsw i32 %N.addr.09, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @sgt_extra_use_cmp(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_extra_use_cmp(
; COMMON:       vector.body:
; CHECK-TF:     masked.load
; CHECK-TF:     masked.load
; CHECK-TF:     masked.store
entry:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ %N, %entry ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %entry ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %entry ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %entry ]
  %dec = add nsw i32 %N.addr.09, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  %select = select i1 %cmp, i8 %0, i8 %1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @sgt_const_tripcount(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_const_tripcount(
; COMMON:       vector.body:
; CHECK-TF:     masked.load
; CHECK-TF:     masked.load
; CHECK-TF:     masked.store
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %while.body.preheader, label %while.end

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ 2049, %while.body.preheader ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %while.body.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %while.body.preheader ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %while.body.preheader ]
  %dec = add nsw i32 %N.addr.09, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @sgt_no_guard_0_startval(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_no_guard_0_startval(
; COMMON-NOT:   vector.body:
entry:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ 0, %entry ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %entry ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %entry ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %entry]
  %dec = add nsw i32 %N.addr.09, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @sgt_step_minus_two(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL:  @sgt_step_minus_two(
; COMMON:        vector.body:
; CHECK-TF:      masked.load
; CHECK-TF:      masked.load
; CHECK-TF:      masked.store
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %while.body.preheader, label %while.end

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %while.body.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %while.body.preheader ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %while.body.preheader ]
  %dec = add nsw i32 %N.addr.09, -2
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @sgt_step_not_constant(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N, i32 %S) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_step_not_constant(
; COMMON-NOT:   vector.body:
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %while.body.preheader, label %while.end

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.09 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %c.addr.08 = phi ptr [ %incdec.ptr4, %while.body ], [ %c, %while.body.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr1, %while.body ], [ %b, %while.body.preheader ]
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %while.body.preheader ]
  %dec = add nsw i32 %N.addr.09, %S
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %0 = load i8, ptr %a.addr.06, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %b.addr.07, i32 1
  %1 = load i8, ptr %b.addr.07, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %c.addr.08, i32 1
  store i8 %add, ptr %c.addr.08, align 1
  %cmp = icmp sgt i32 %N.addr.09, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

define dso_local void @icmp_eq(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) #0 {
; COMMON-LABEL: @icmp_eq
; COMMON:       vector.body:
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.010 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %C.addr.09 = phi ptr [ %incdec.ptr4, %while.body ], [ %C, %while.body.preheader ]
  %B.addr.08 = phi ptr [ %incdec.ptr1, %while.body ], [ %B, %while.body.preheader ]
  %A.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %A, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, ptr %A.addr.07, i32 1
  %0 = load i8, ptr %A.addr.07, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %B.addr.08, i32 1
  %1 = load i8, ptr %B.addr.08, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %C.addr.09, i32 1
  store i8 %add, ptr %C.addr.09, align 1
  %dec = add i32 %N.addr.010, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

; This IR corresponds to this type of C-code:
;
;  void f(char *a, char *b, char * __restrict c, int N) {
;    #pragma clang loop vectorize_width(16)
;    for (int i = N; i>0; i--)
;      c[i] = a[i] + b[i];
;  }
;
define dso_local void @sgt_for_loop(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_for_loop(
; COMMON:       vector.body:
; CHECK-PREFER: masked.load
; CHECK-PREFER: masked.load
; CHECK-PREFER: masked.store
;
; TODO: if tail-predication is requested, tail-folding isn't triggered because
; the profitability check returns "Different strides found, can't tail-predicate",
; investigate this.
;
; CHECK-ENABLE-TP-NOT: masked.load
; CHECK-ENABLE-TP-NOT: masked.load
; CHECK-ENABLE-TP-NOT: masked.store
;
entry:
  %cmp5 = icmp sgt i32 %N, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i.011 = phi i32 [ %dec, %for.body ], [ %N, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 %i.011
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %b, i32 %i.011
  %1 = load i8, ptr %arrayidx1, align 1
  %add = add i8 %1, %0
  %arrayidx4 = getelementptr inbounds i8, ptr %c, i32 %i.011
  store i8 %add, ptr %arrayidx4, align 1
  %dec = add nsw i32 %i.011, -1
  %cmp = icmp sgt i32 %i.011, 1
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !1

for.end:
  ret void
}

define dso_local void @sgt_for_loop_i64(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_for_loop_i64(
; COMMON:       vector.body:
;
; CHECK-PREFER: masked.load
; CHECK-PREFER: masked.load
; CHECK-PREFER: masked.store
;
; With -disable-mve-tail-predication=false, the target hook returns
; "preferPredicateOverEpilogue: hardware-loop is not profitable."
; so here we don't expect the tail-folding. TODO: look into this.
;
; CHECK-ENABLE-TP-NOT:  masked.load
; CHECK-ENABLE-TP-NOT:  masked.load
; CHECK-ENABLE-TP-NOT:  masked.store
;
entry:
  %cmp14 = icmp sgt i32 %N, 0
  br i1 %cmp14, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %conv16 = zext i32 %N to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.015 = phi i64 [ %dec, %for.body ], [ %conv16, %for.body.preheader ]
  %idxprom = trunc i64 %i.015 to i32
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 %idxprom
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx4 = getelementptr inbounds i8, ptr %b, i32 %idxprom
  %1 = load i8, ptr %arrayidx4, align 1
  %add = add i8 %1, %0
  %arrayidx8 = getelementptr inbounds i8, ptr %c, i32 %idxprom
  store i8 %add, ptr %arrayidx8, align 1
  %dec = add nsw i64 %i.015, -1
  %cmp = icmp sgt i64 %i.015, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !1
}

; This IR corresponds to this nested-loop:
;
;   for (int i = 0; i<N; i++)
;     for (int j = i+1; j>0; j--)
;       c[j] = a[j] + b[j];
;
; while the inner-loop looks similar to previous examples, we can't
; transform this because the inner loop because isGuarded returns
; false for the inner-loop.
;
define dso_local void @sgt_nested_loop(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: @sgt_nested_loop(
; DEFAULT-NOT:  vector.body:
; CHECK-TF-NOT: masked.load
; CHECK-TF-NOT: masked.load
; CHECK-TF-NOT: masked.store
; COMMON:       }
;
entry:
  %cmp21 = icmp sgt i32 %N, 0
  br i1 %cmp21, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.loopexit:
  %exitcond = icmp eq i32 %add, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.022 = phi i32 [ %add, %for.cond.loopexit ], [ 0, %for.body.preheader ]
  %add = add nuw nsw i32 %i.022, 1
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %j.020 = phi i32 [ %add, %for.body ], [ %dec, %for.body4 ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 %j.020
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx5 = getelementptr inbounds i8, ptr %b, i32 %j.020
  %1 = load i8, ptr %arrayidx5, align 1
  %add7 = add i8 %1, %0
  %arrayidx9 = getelementptr inbounds i8, ptr %c, i32 %j.020
  store i8 %add7, ptr %arrayidx9, align 1
  %dec = add nsw i32 %j.020, -1
  %cmp2 = icmp sgt i32 %j.020, 1
  br i1 %cmp2, label %for.body4, label %for.cond.loopexit
}

attributes #0 = { nofree norecurse nounwind "target-features"="+armv8.1-m.main,+mve.fp" }

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.width", i32 16}
