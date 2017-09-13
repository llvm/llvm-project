; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -enable-if-conversion -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This is the loop in this example:
;
;int function0(int *a, int *b, int start, int end) {
;
;  for (int i=start; i<end; ++i) {
;    unsigned k = a[i];
;
;    if (a[i] > b[i])   <------ notice the IF inside the loop.
;      k = k * 5 + 3;
;
;    a[i] = k;  <---- K is a phi node that becomes vector-select.
;  }
;}

;CHECK-LABEL: @function0(
;CHECK: load <4 x i32>
;CHECK: icmp sgt <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: add <4 x i32>
;CHECK: select <4 x i1>
;CHECK: ret i32
define i32 @function0(i32* nocapture %a, i32* nocapture %b, i32 %start, i32 %end) nounwind uwtable ssp {
entry:
  %cmp16 = icmp slt i32 %start, %end
  br i1 %cmp16, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = sext i32 %start to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %0, %for.body.lr.ph ], [ %indvars.iv.next, %if.end ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx4, align 4
  %cmp5 = icmp sgt i32 %1, %2
  br i1 %cmp5, label %if.then, label %if.end

if.then:
  %mul = mul i32 %1, 5
  %add = add i32 %mul, 3
  br label %if.end

if.end:
  %k.0 = phi i32 [ %add, %if.then ], [ %1, %for.body ]
  store i32 %k.0, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %3 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %3, %end
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 undef
}



; int func(int *A, int n) {
;   unsigned sum = 0;
;   for (int i = 0; i < n; ++i)
;     if (A[i] > 30)
;       sum += A[i] + 2;
;
;   return sum;
; }

;CHECK-LABEL: @reduction_func(
;CHECK: load <4 x i32>
;CHECK: icmp slt <4 x i32>
;CHECK: add <4 x i32>
;CHECK: select <4 x i1>
;CHECK: ret i32
define i32 @reduction_func(i32* nocapture %A, i32 %n) nounwind uwtable readonly ssp {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %sum.011 = phi i32 [ %sum.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 30
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %add = add i32 %sum.011, 2
  %add4 = add i32 %add, %0
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %sum.1 = phi i32 [ %add4, %if.then ], [ %sum.011, %for.body ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %sum.1, %for.inc ]
  ret i32 %sum.0.lcssa
}

@a = common global [1 x i32*] zeroinitializer, align 8
@c = common global i32* null, align 8

; We use to if convert this loop. This is not safe because there is a trapping
; constant expression.
; PR16729

; CHECK-LABEL: trapping_constant_expression
; CHECK-NOT: or <4 x i32>

define i32 @trapping_constant_expression() {
entry:
  br label %for.body

for.body:
  %inc3 = phi i32 [ 0, %entry ], [ %inc, %cond.end ]
  %or2 = phi i32 [ 0, %entry ], [ %or, %cond.end ]
  br i1 icmp eq (i32** getelementptr inbounds ([1 x i32*], [1 x i32*]* @a, i64 0, i64 0), i32** @c), label %cond.false, label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond = phi i32 [ sdiv (i32 1, i32 zext (i1 icmp eq (i32** getelementptr inbounds ([1 x i32*], [1 x i32*]* @a, i64 0, i64 0), i32** @c) to i32)), %cond.false ], [ 0, %for.body ]
  %or = or i32 %or2, %cond
  %inc = add nsw i32 %inc3, 1
  %cmp = icmp slt i32 %inc, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 %or
}

; Neither should we if-convert if there is an instruction operand that is a
; trapping constant expression.
; PR16729

; CHECK-LABEL: trapping_constant_expression2
; CHECK-NOT: or <4 x i32>

define i32 @trapping_constant_expression2() {
entry:
  br label %for.body

for.body:
  %inc3 = phi i32 [ 0, %entry ], [ %inc, %cond.end ]
  %or2 = phi i32 [ 0, %entry ], [ %or, %cond.end ]
  br i1 icmp eq (i32** getelementptr inbounds ([1 x i32*], [1 x i32*]* @a, i64 0, i64 0), i32** @c), label %cond.false, label %cond.end

cond.false:
  %cond.1 = or i32 %inc3, sdiv (i32 1, i32 zext (i1 icmp eq (i32** getelementptr inbounds ([1 x i32*], [1 x i32*]* @a, i64 0, i64 1), i32** @c) to i32))
  br label %cond.end

cond.end:
  %cond = phi i32 [ %cond.1, %cond.false ], [ %inc3, %for.body ]
  %or = or i32 %or2, %cond
  %inc = add nsw i32 %inc3, 1
  %cmp = icmp slt i32 %inc, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 %or
}

; Handle PHI with single incoming value having a full mask.
; PR34523

; CHECK-LABEL: PR34523
; CHECK: vector.body

define void @PR34523() {
bb1:
  br label %bb2

bb2:                                             ; preds = %bb4, %bb1
  %i = phi i16 [ undef, %bb1 ], [ %_tmp2, %bb4 ]
  br label %bb3

bb3:                                             ; preds = %bb2
  %_tmp1 = phi [1 x [1 x i32]]* [ undef, %bb2 ]
  br label %bb4

bb4:                                             ; preds = %bb3
  %_tmp2 = add i16 %i, 1
  %_tmp3 = icmp slt i16 %_tmp2, 2
  br i1 %_tmp3, label %bb2, label %bb5

bb5:                                             ; preds = %bb4
  unreachable
}
