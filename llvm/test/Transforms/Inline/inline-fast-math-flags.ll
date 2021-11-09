; RUN: opt < %s -S -inline -inline-threshold=20 | FileCheck %s
; RUN: opt < %s -S -passes='cgscc(inline)' -inline-threshold=20 | FileCheck %s
; RUN: opt < %s -S -passes='module-inline' -inline-threshold=20 | FileCheck %s
; Check that we don't drop FastMathFlag when estimating inlining profitability.
;
; In this test we should inline 'foo'  to 'boo', because it'll fold to a
; constant.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float* %a, float %b) {
entry:
  %a0 = load float, float* %a, align 4
  %mul = fmul fast float %a0, %b
  %tobool = fcmp une float %mul, 0.000000e+00
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %a1 = load float, float* %a, align 8
  %arrayidx1 = getelementptr inbounds float, float* %a, i64 1
  %a2 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %a1, %a2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %storemerge = phi float [ %add, %if.then ], [ 1.000000e+00, %entry ]
  ret float %storemerge
}

; CHECK-LABEL: @boo
; CHECK-NOT: call float @foo
define float @boo(float* %a) {
entry:
  %call = call float @foo(float* %a, float 0.000000e+00)
  ret float %call
}
