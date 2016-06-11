; RUN: opt -S -debug-only=loop-vectorize -loop-vectorize -instcombine < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnueabi"

@AB = common global [1024 x i8] zeroinitializer, align 4
@CD = common global [1024 x i8] zeroinitializer, align 4

define void @test_byte_interleaved_cost(i8 %C, i8 %D) {
entry:
  br label %for.body

; 8xi8 and 16xi8 are valid i8 vector types, so the cost of the interleaved
; access group is 2.

; CHECK: LV: Checking a loop in "test_byte_interleaved_cost"
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %tmp = load i8, i8* %arrayidx0, align 4
; CHECK: LV: Found an estimated cost of 2 for VF 16 For instruction:   %tmp = load i8, i8* %arrayidx0, align 4

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx0 = getelementptr inbounds [1024 x i8], [1024 x i8]* @AB, i64 0, i64 %indvars.iv
  %tmp = load i8, i8* %arrayidx0, align 4
  %tmp1 = or i64 %indvars.iv, 1
  %arrayidx1 = getelementptr inbounds [1024 x i8], [1024 x i8]* @AB, i64 0, i64 %tmp1
  %tmp2 = load i8, i8* %arrayidx1, align 4
  %add = add nsw i8 %tmp, %C
  %mul = mul nsw i8 %tmp2, %D
  %arrayidx2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @CD, i64 0, i64 %indvars.iv
  store i8 %add, i8* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [1024 x i8], [1024 x i8]* @CD, i64 0, i64 %tmp1
  store i8 %mul, i8* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

%ig.factor.8 = type { double*, double, double, double, double, double, double, double }
define double @wide_interleaved_group(%ig.factor.8* %s, double %a, double %b, i32 %n) {
entry:
  br label %for.body

; Check the default cost of a strided load with a factor that is greater than
; the maximum allowed. In this test, the interleave factor would be 8, which is
; not supported.

; CHECK: LV: Checking a loop in "wide_interleaved_group"
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction:   %1 = load double, double* %0, align 8
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %5 = load double, double* %4, align 8
; CHECK: LV: Found an estimated cost of 10 for VF 2 For instruction:   store double %9, double* %10, align 8

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %r = phi double [ 0.000000e+00, %entry ], [ %12, %for.body ]
  %0 = getelementptr inbounds %ig.factor.8, %ig.factor.8* %s, i64 %i, i32 2
  %1 = load double, double* %0, align 8
  %2 = fcmp fast olt double %1, %a
  %3 = select i1 %2, double 0.000000e+00, double %1
  %4 = getelementptr inbounds %ig.factor.8, %ig.factor.8* %s, i64 %i, i32 6
  %5 = load double, double* %4, align 8
  %6 = fcmp fast olt double %5, %a
  %7 = select i1 %6, double 0.000000e+00, double %5
  %8 = fmul fast double %7, %b
  %9 = fadd fast double %8, %3
  %10 = getelementptr inbounds %ig.factor.8, %ig.factor.8* %s, i64 %i, i32 3
  store double %9, double* %10, align 8
  %11 = fmul fast double %9, %9
  %12 = fadd fast double %11, %r
  %i.next = add nuw nsw i64 %i, 1
  %13 = trunc i64 %i.next to i32
  %cond = icmp eq i32 %13, %n
  br i1 %cond, label %for.exit, label %for.body

for.exit:
  %r.lcssa = phi double [ %12, %for.body ]
  ret double %r.lcssa
}
