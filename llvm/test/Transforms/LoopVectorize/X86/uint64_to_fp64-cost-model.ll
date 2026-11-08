; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -S -debug-only=loop-vectorize 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"


; CHECK: cost of 4 for VF 1 For instruction:   %conv = uitofp i64 %tmp to double
; CHECK: Cost of 5 for VF 2: WIDEN-CAST ir<%conv> = uitofp ir<%tmp> to double
; CHECK: Cost of 10 for VF 4: WIDEN-CAST ir<%conv> = uitofp ir<%tmp> to double
define void @uint64_to_double_cost(ptr noalias nocapture %a, ptr noalias nocapture readonly %b) {
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %tmp = load i64, ptr %arrayidx, align 4
  %conv = uitofp i64 %tmp to double
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  store double %conv, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in 'fp_induction_i64_index_cost'
; CHECK: Cost of 1 for VF 2: vp<%{{[0-9]+}}> = DERIVED-IV ir<0.000000e+00> + vp<%{{[0-9]+}}> * ir<1.000000e+00>
; CHECK: Cost of 1 for VF 4: vp<%{{[0-9]+}}> = DERIVED-IV ir<0.000000e+00> + vp<%{{[0-9]+}}> * ir<1.000000e+00>
; CHECK: Cost of 1 for VF 8: vp<%{{[0-9]+}}> = DERIVED-IV ir<0.000000e+00> + vp<%{{[0-9]+}}> * ir<1.000000e+00>
define void @fp_induction_i64_index_cost(ptr noalias %a, ptr noalias %b, ptr noalias %d, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %fp.iv = phi float [ 0.000000e+00, %entry ], [ %fp.iv.next, %loop ]
  %gep.b = getelementptr inbounds float, ptr %b, i64 %iv
  %l = load float, ptr %gep.b, align 4
  %add = fadd fast float %l, 1.000000e+00
  %gep.a = getelementptr inbounds float, ptr %a, i64 %iv
  store float %add, ptr %gep.a, align 4
  %idx = mul nuw nsw i64 %iv, 2
  %gep.d = getelementptr inbounds float, ptr %d, i64 %idx
  store float %fp.iv, ptr %gep.d, align 4
  %fp.iv.next = fadd fast float %fp.iv, 1.000000e+00
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
