; RUN: opt -S -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 < %s | FileCheck %s -check-prefix=VF8
; RUN: opt -S -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=4 < %s | FileCheck %s -check-prefix=VF1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Given a loop with an induction variable which is being
; truncated/extended using casts that had been proven to
; be redundant under a runtime test, we want to make sure
; that these casts, do not get vectorized/scalarized/widened. 
; This is the case for inductions whose SCEV expression is
; of the form "ExtTrunc(%phi) + %step", where "ExtTrunc"
; can be a result of the IR sequences we check below.
; 
; See also pr30654.
;

; Case1: Check the following induction pattern:
;
;  %p.09 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
;  %sext = shl i32 %p.09, 24
;  %conv = ashr exact i32 %sext, 24
;  %add = add nsw i32 %conv, %step
; 
; This is the case in the following code:
;
; void doit1(int n, int step) {
;   int i;
;   char p = 0;
;   for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + step;
;   }
; }
;
; The "ExtTrunc" IR sequence here is:
;  "%sext = shl i32 %p.09, 24"
;  "%conv = ashr exact i32 %sext, 24"
; We check that it does not appear in the vector loop body, whether
; we vectorize or scalarize the induction.
; In the case of widened induction, this means that the induction phi
; is directly used, without shl/ashr on the way.

; VF8-LABEL: @doit1
; VF8: vector.body:
; VF8: %vec.ind = phi <8 x i32>
; VF8: store <8 x i32> %vec.ind
; VF8: middle.block:            

; VF1-LABEL: @doit1
; VF1: vector.body:
; VF1-NOT: %{{.*}} = shl i32
; VF1: middle.block:            

@a = common local_unnamed_addr global [250 x i32] zeroinitializer, align 16

define void @doit1(i32 %n, i32 %step) {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %p.09 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %sext = shl i32 %p.09, 24
  %conv = ashr exact i32 %sext, 24
  %arrayidx = getelementptr inbounds [250 x i32], ptr @a, i64 0, i64 %indvars.iv
  store i32 %conv, ptr %arrayidx, align 4
  %add = add nsw i32 %conv, %step
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; Case2: Another variant of the above pattern is where the induction variable
; is used only for address compuation (i.e. it is a GEP index) and therefore
; the induction is not vectorized but rather only the step is widened. 
;
; This is the case in the following code, where the induction variable 'w_ix' 
; is only used to access the array 'in':
;
; void doit2(int *in, int *out, size_t size, size_t step)
; {
;    int w_ix = 0;
;    for (size_t offset = 0; offset < size; ++offset)
;     {
;        int w = in[w_ix];
;        out[offset] = w;
;        w_ix += step;
;     }
; }
;
; The "ExtTrunc" IR sequence here is similar to the previous case:
;  "%sext = shl i64 %w_ix.012, 32
;  %idxprom = ashr exact i64 %sext, 32"
; We check that it does not appear in the vector loop body, whether
; we widen or scalarize the induction.
; In the case of widened induction, this means that the induction phi
; is directly used, without shl/ashr on the way.

; VF8-LABEL: @doit2
; VF8: vector.body:
; VF8-NEXT:  [[INDEX:%.+]] = phi i64 [ 0, %vector.ph ]
; VF8-NEXT:  [[OFFSET_IDX:%.+]] = mul i64 [[INDEX]], %step
; VF8-NEXT:  [[MUL0:%.+]] = mul i64 0, %step
; VF8-NEXT:  [[ADD:%.+]] = add i64 [[OFFSET_IDX]], [[MUL0]]
; VF8:       [[I0:%.+]] = add i64 [[INDEX]], 0
; VF8:       getelementptr inbounds i32, ptr %in, i64 [[ADD]]
; VF8: middle.block:

; VF1-LABEL: @doit2
; VF1: vector.body:
; VF1-NOT: %{{.*}} = shl i64
; VF1: middle.block:
;

define void @doit2(ptr nocapture readonly %in, ptr nocapture %out, i64 %size, i64 %step)  {
entry:
  %cmp9 = icmp eq i64 %size, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %w_ix.011 = phi i64 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %offset.010 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %sext = shl i64 %w_ix.011, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %out, i64 %offset.010
  store i32 %0, ptr %arrayidx1, align 4
  %add = add i64 %idxprom, %step
  %inc = add nuw i64 %offset.010, 1
  %exitcond = icmp eq i64 %inc, %size
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; Case3: Lastly, check also the following induction pattern:
; 
;  %p.09 = phi i32 [ %val0, %scalar.ph ], [ %add, %for.body ]
;  %conv = and i32 %p.09, 255
;  %add = add nsw i32 %conv, %step
; 
; This is the case in the following code:
;
; int a[N];
; void doit3(int n, int step) {
;   int i;
;   unsigned char p = 0;
;   for (i = 0; i < n; i++) {
;      a[i] = p;
;      p = p + step;
;   }
; }
; 
; The "ExtTrunc" IR sequence here is:
;  "%conv = and i32 %p.09, 255".
; We check that it does not appear in the vector loop body, whether
; we vectorize or scalarize the induction.

; VF8-LABEL: @doit3
; VF8: vector.body:
; VF8: %vec.ind = phi <8 x i32>
; VF8: store <8 x i32> %vec.ind
; VF8: middle.block:            

; VF1-LABEL: @doit3
; VF1: vector.body:
; VF1-NOT: %{{.*}} = and i32 
; VF1: middle.block:            

define void @doit3(i32 %n, i32 %step) {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %p.09 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %conv = and i32 %p.09, 255
  %arrayidx = getelementptr inbounds [250 x i32], ptr @a, i64 0, i64 %indvars.iv
  store i32 %conv, ptr %arrayidx, align 4
  %add = add nsw i32 %conv, %step
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; VF8-LABEL: @test_conv_in_latch_block
; VF8: vector.body:
; VF8-NEXT: %index = phi i64
; VF8-NEXT: %vec.ind = phi <8 x i32>
; VF8: store <8 x i32> %vec.ind
; VF8: middle.block:
;
define void @test_conv_in_latch_block(i32 %n, i32 %step, ptr noalias %A, ptr noalias %B) {
entry:
  %wide.trip.count = zext i32 %n to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %p.09 = phi i32 [ 0, %entry ], [ %add, %latch ]
  %B.gep = getelementptr inbounds i32, ptr %B, i64 %iv
  %l = load i32, ptr %B.gep
  %c = icmp eq i32 %l, 0
  br i1 %c, label %then, label %latch

then:
  %A.gep = getelementptr inbounds i32, ptr %A, i64 %iv
  store i32 0, ptr %A.gep
  br label %latch

latch:
  %sext = shl i32 %p.09, 24
  %conv = ashr exact i32 %sext, 24
  %add = add nsw i32 %conv, %step
  store i32 %conv, ptr %B.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
