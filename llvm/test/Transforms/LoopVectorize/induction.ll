; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure that we can handle multiple integer induction variables.
; CHECK-LABEL: @multi_int_induction(
; CHECK: vector.body:
; CHECK:  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:  %[[VAR:.*]] = trunc i64 %index to i32
; CHECK:  %offset.idx = add i32 190, %[[VAR]]
define void @multi_int_induction(i32* %A, i32 %N) {
for.body.lr.ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %count.09 = phi i32 [ 190, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %count.09, i32* %arrayidx2, align 4
  %inc = add nsw i32 %count.09, 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -instcombine -S | FileCheck %s --check-prefix=IND

; Make sure we remove unneeded vectorization of induction variables.
; In order for instcombine to cleanup the vectorized induction variables that we
; create in the loop vectorizer we need to perform some form of redundancy
; elimination to get rid of multiple uses.

; IND-LABEL: scalar_use

; IND:     br label %vector.body
; IND:     vector.body:
;   Vectorized induction variable.
; IND-NOT:  insertelement <2 x i64>
; IND-NOT:  shufflevector <2 x i64>
; IND:     br {{.*}}, label %vector.body

define void @scalar_use(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float, float* %a, i64 %ind.sum
  %l1 = load float, float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float, float* %a, i64 %ind.sum2
  %l2 = load float, float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}


; Make sure that the loop exit count computation does not overflow for i8 and
; i16. The exit count of these loops is i8/i16 max + 1. If we don't cast the
; induction variable to a bigger type the exit count computation will overflow
; to 0.
; PR17532

; CHECK-LABEL: i8_loop
; CHECK: icmp eq i32 {{.*}}, 256
define i32 @i8_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i8 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i8 %b.0, -1
  %4 = icmp eq i8 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; CHECK-LABEL: i16_loop
; CHECK: icmp eq i32 {{.*}}, 65536

define i32 @i16_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i16 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i16 %b.0, -1
  %4 = icmp eq i16 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; This loop has a backedge taken count of i32_max. We need to check for this
; condition and branch directly to the scalar loop.

; CHECK-LABEL: max_i32_backedgetaken
; CHECK:  br i1 true, label %scalar.ph, label %min.iters.checked

; CHECK: scalar.ph:
; CHECK:  %bc.resume.val = phi i32 [ 0, %middle.block ], [ 0, %0 ]
; CHECK:  %bc.merge.rdx = phi i32 [ 1, %0 ], [ 1, %min.iters.checked ], [ %5, %middle.block ]

define i32 @max_i32_backedgetaken() nounwind readnone ssp uwtable {

  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i32 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i32 %b.0, -1
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; When generating the overflow check we must sure that the induction start value
; is defined before the branch to the scalar preheader.

; CHECK-LABEL: testoverflowcheck
; CHECK: entry
; CHECK: %[[LOAD:.*]] = load i8
; CHECK: br

; CHECK: scalar.ph
; CHECK: phi i8 [ %{{.*}}, %middle.block ], [ %[[LOAD]], %entry ]

@e = global i8 1, align 1
@d = common global i32 0, align 4
@c = common global i32 0, align 4
define i32 @testoverflowcheck() {
entry:
  %.pr.i = load i8, i8* @e, align 1
  %0 = load i32, i32* @d, align 4
  %c.promoted.i = load i32, i32* @c, align 4
  br label %cond.end.i

cond.end.i:
  %inc4.i = phi i8 [ %.pr.i, %entry ], [ %inc.i, %cond.end.i ]
  %and3.i = phi i32 [ %c.promoted.i, %entry ], [ %and.i, %cond.end.i ]
  %and.i = and i32 %0, %and3.i
  %inc.i = add i8 %inc4.i, 1
  %tobool.i = icmp eq i8 %inc.i, 0
  br i1 %tobool.i, label %loopexit, label %cond.end.i

loopexit:
  ret i32 %and.i
}

; The SCEV expression of %sphi is (zext i8 {%t,+,1}<%loop> to i32)
; In order to recognize %sphi as an induction PHI and vectorize this loop,
; we need to convert the SCEV expression into an AddRecExpr.
; The expression gets converted to {zext i8 %t to i32,+,1}.

; CHECK-LABEL: wrappingindvars1
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
; CHECK: add <2 x i32> {{%[^ ]*}}, <i32 0, i32 1>
define void @wrappingindvars1(i8 %t, i32 %len, i32 *%A) {
 entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %idx.b = phi i32 [ 0, %entry ], [ %idx.b.inc, %loop ]
  %sphi = phi i32 [ %ext, %entry ], [%idx.inc.ext, %loop]

  %ptr = getelementptr inbounds i32, i32* %A, i8 %idx
  store i32 %sphi, i32* %ptr

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %idx.b.inc = add nuw nsw i32 %idx.b, 1

  %c = icmp ult i32 %idx.b, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

; The SCEV expression of %sphi is (4 * (zext i8 {%t,+,1}<%loop> to i32))
; In order to recognize %sphi as an induction PHI and vectorize this loop,
; we need to convert the SCEV expression into an AddRecExpr.
; The expression gets converted to ({4 * (zext %t to i32),+,4}).
; CHECK-LABEL: wrappingindvars2
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.body
; CHECK: add <2 x i32> {{%[^ ]*}}, <i32 0, i32 4>
define void @wrappingindvars2(i8 %t, i32 %len, i32 *%A) {

entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ext.mul = mul i32 %ext, 4

  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %sphi = phi i32 [ %ext.mul, %entry ], [%mul, %loop]
  %idx.b = phi i32 [ 0, %entry ], [ %idx.b.inc, %loop ]

  %ptr = getelementptr inbounds i32, i32* %A, i8 %idx
  store i32 %sphi, i32* %ptr

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %mul = mul i32 %idx.inc.ext, 4
  %idx.b.inc = add nuw nsw i32 %idx.b, 1

  %c = icmp ult i32 %idx.b, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}
