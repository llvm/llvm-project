; RUN: opt -indvars -loop-vectorize -force-vector-width=2 -force-vector-unroll=1 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; We must not vectorize this loop. %add55 is not reduction. Its value is used
; multiple times.

; PR18526

; CHECK: multiple_use_of_value
; CHECK-NOT: <2 x i32>

define void @multiple_use_of_value() {
entry:
  %n = alloca i32, align 4
  %k7 = alloca i32, align 4
  %nf = alloca i32, align 4
  %0 = load i32* %k7, align 4
  %.neg1 = sub i32 0, %0
  %n.promoted = load i32* %n, align 4
  %nf.promoted = load i32* %nf, align 4
  br label %for.body

for.body:
  %inc107 = phi i32 [ undef, %entry ], [ %inc10, %for.body ]
  %inc6 = phi i32 [ %nf.promoted, %entry ], [ undef, %for.body ]
  %add55 = phi i32 [ %n.promoted, %entry ], [ %add5, %for.body ]
  %.neg2 = sub i32 0, %inc6
  %add.neg = add i32 0, %add55
  %add4.neg = add i32 %add.neg, %.neg1
  %sub = add i32 %add4.neg, %.neg2
  %add5 = add i32 %sub, %add55
  %inc10 = add i32 %inc107, 1
  %cmp = icmp ult i32 %inc10, 61
  br i1 %cmp, label %for.body, label %for.end

for.end:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  store i32 %add5.lcssa, i32* %n, align 4
  ret void
}
