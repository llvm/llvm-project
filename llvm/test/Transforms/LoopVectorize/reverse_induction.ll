; RUN: opt < %s -loop-vectorize -force-vector-unroll=2 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure consecutive vector generates correct negative indices.
; PR15882

; CHECK: reverse_induction_i64
; CHECK: add <4 x i64> %[[SPLAT:.*]], <i64 0, i64 -1, i64 -2, i64 -3>
; CHECK: add <4 x i64> %[[SPLAT]], <i64 -4, i64 -5, i64 -6, i64 -7>

define i32 @reverse_induction_i64(i64 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i64 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i64 %add.i7, -1
  %kind_.i = getelementptr inbounds i32* %ptr, i64 %add.i
  %tmp.i1 = load i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}

; CHECK: reverse_induction_i128
; CHECK: add <4 x i128> %[[SPLAT:.*]], <i128 0, i128 -1, i128 -2, i128 -3>
; CHECK: add <4 x i128> %[[SPLAT]], <i128 -4, i128 -5, i128 -6, i128 -7>
define i32 @reverse_induction_i128(i128 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i128 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i128 %add.i7, -1
  %kind_.i = getelementptr inbounds i32* %ptr, i128 %add.i
  %tmp.i1 = load i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}

; CHECK: reverse_induction_i16
; CHECK: add <4 x i16> %[[SPLAT:.*]], <i16 0, i16 -1, i16 -2, i16 -3>
; CHECK: add <4 x i16> %[[SPLAT]], <i16 -4, i16 -5, i16 -6, i16 -7>

define i32 @reverse_induction_i16(i16 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i16 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i16 %add.i7, -1
  %kind_.i = getelementptr inbounds i32* %ptr, i16 %add.i
  %tmp.i1 = load i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}


