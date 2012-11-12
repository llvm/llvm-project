; RUN: opt < %s -S -indvars -loop-unroll | FileCheck %s
;
; loop-unroll fully unrolls the inner loop, creating an interesting
; chain of multiplication. indvars forces SCEV to run again on the
; outer loop. While reducing the recurrence at %mul3, unsigned integer overflow
; causes one of the terms to reach zero. This forces all multiples in
; the recurrence to be zero, reducing the whole thing to a constant expression.
;
; PR12929: cast<Ty>() argument of incompatible type

; CHECK: @func
; CHECK: for.cond:
; CHECK: %inc1 = phi i8 [ 0, %entry ], [ %0, %for.body ]
; CHECK: br label %for.body

; CHECK: for.body:
; CHECK: %inc.9 = add i8 %inc.8, 1
; CHECK: %0 = add i8 %inc1, 10
; CHECK: br label %for.cond

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
define void @func() noreturn nounwind uwtable ssp {
entry:
  br label %for.cond

for.cond.loopexit:                                ; preds = %for.body
  %mul.lcssa = phi i8 [ %mul, %for.body ]
  %0 = add i8 %inc1, 10
  %indvars.iv.next = add i8 %indvars.iv, 10
  br label %for.cond

for.cond:                                         ; preds = %for.cond.loopexit, %entry
  %indvars.iv = phi i8 [ %indvars.iv.next, %for.cond.loopexit ], [ 10, %entry ]
  %mul3 = phi i8 [ undef, %entry ], [ %mul.lcssa, %for.cond.loopexit ]
  %inc1 = phi i8 [ 0, %entry ], [ %0, %for.cond.loopexit ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.cond
  %inc26 = phi i8 [ %inc1, %for.cond ], [ %inc, %for.body ]
  %mul45 = phi i8 [ %mul3, %for.cond ], [ %mul, %for.body ]
  %inc = add i8 %inc26, 1
  %mul = mul i8 %inc26, %mul45
  %exitcond = icmp ne i8 %inc, %indvars.iv
  br i1 %exitcond, label %for.body, label %for.cond.loopexit
}
