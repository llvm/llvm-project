; RUN: opt -loop-vectorize -force-vector-width=2 -pass-remarks-missed='loop-vectorize' -S < %s 2>&1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: loop not vectorized: integer loop induction variable could not be identified

; Test-case ('-O2 -ffast-math') from PR38800.
; (Set '-force-vector-width=2' to enable vector code generation.)
;
; No integral induction variable in the source-code caused a compiler-crash
; when attempting to vectorize.  With the fix, a remark indicating why it
; wasn't vectorized is produced
;
;void foo(float *ptr, float val) {
;  float f;
;  for (f = 0.1f; f < 1.0f; f += 0.01f)
;    *ptr += val;
;}

define void @foo(float* nocapture %ptr, float %val) local_unnamed_addr {
entry:
  %ptr.promoted = load float, float* %ptr, align 4
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %add5 = phi float [ %ptr.promoted, %entry ], [ %add, %for.body ]
  %f.04 = phi float [ 0x3FB99999A0000000, %entry ], [ %add1, %for.body ]
  %add = fadd fast float %add5, %val
  %add1 = fadd fast float %f.04, 0x3F847AE140000000
  %cmp = fcmp fast olt float %add1, 1.000000e+00
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  store float %add, float* %ptr, align 4
  ret void
}
