; RUN: opt < %s -enable-interleaved-mem-accesses=true -force-vector-width=4 -loop-vectorize -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

; Verify that we do not get any loads of vectors with x86_fp80 elements.
;
; CHECK-NOT: load {{.*}} x x86_fp80
define x86_fp80 @foo(x86_fp80* %a) {
entry:
  br label %for.body

for.cond.cleanup:
  ret x86_fp80 %3

for.body:
  %i.09 = phi i16 [ 0, %entry ], [ %add3, %for.body ]
  %res.08 = phi x86_fp80 [ undef, %entry ], [ %3, %for.body ]
  %arrayidx = getelementptr inbounds x86_fp80, x86_fp80* %a, i16 %i.09
  %0 = load x86_fp80, x86_fp80* %arrayidx, align 1
  %add = or i16 %i.09, 1
  %arrayidx2 = getelementptr inbounds x86_fp80, x86_fp80* %a, i16 %add
  %1 = load x86_fp80, x86_fp80* %arrayidx2, align 1
  %2 = fadd fast x86_fp80 %0, %res.08
  %3 = fadd fast x86_fp80 %2, %1
  %add3 = add nuw nsw i16 %i.09, 2
  %cmp = icmp ult i16 %add3, 400
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
