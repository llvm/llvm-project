; RUN: opt < %s -mtriple=powerpc64le-unknown-linux-gnu -simplifycfg -hoist-common-insts=true -S | FileCheck %s

define float @foo(float* %src, float* %dest, i32 signext %count, i32 signext %cond) {
; CHECK-LABEL: @foo(
; CHECK-LABEL: entry:
; CHECK-NOT:  load float
entry:
  %cmp = icmp sgt i32 %cond, 10
  %idxprom = sext i32 %count to i64
  %arrayidx = getelementptr inbounds float, float* %src, i64 %idxprom
  br i1 %cmp, label %if.then, label %if.else

; CHECK-LABEL: if.then:
; CHECK:  %0 = load float, float* %arrayidx, align 4
if.then:                                          ; preds = %entry
  %0 = load float, float* %arrayidx, align 4
  %res = fmul float %0, 3.000000e+00
  br label %if.end

; CHECK-LABEL: if.else:
; CHECK:   %1 = load float, float* %arrayidx, align 4
; CHECK:   store float %1, float* %arrayidx4, align 4
if.else:                                          ; preds = %entry
  %1 = load float, float* %arrayidx, align 4
  %idxprom3 = sext i32 %count to i64
  %arrayidx4 = getelementptr inbounds float, float* %dest, i64 %idxprom3
  store float %1, float* %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %res2.0 = phi float [ %res, %if.then ], [ 0.000000e+00, %if.else ]
  ret float %res2.0
}
