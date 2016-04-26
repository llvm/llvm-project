; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [dim] -> {  : dim > 0 }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [dim] -> {  : dim < 0 }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @horner_bezier_curve(float* %cp, i32 %dim) #0 {
entry:
  br label %for.body18.lr.ph

for.body18.lr.ph:                                 ; preds = %entry
  %add.ptr = getelementptr inbounds float, float* %cp, i64 0
  br label %for.body18

for.body18:                                       ; preds = %for.body18, %for.body18.lr.ph
  %cp.addr.052 = phi float* [ %add.ptr, %for.body18.lr.ph ], [ %add.ptr43, %for.body18 ]
  %arrayidx31 = getelementptr inbounds float, float* %cp.addr.052, i64 0
  %0 = load float, float* %arrayidx31, align 4
  store float %0, float* %arrayidx31, align 4
  %idx.ext42 = zext i32 %dim to i64
  %add.ptr43 = getelementptr inbounds float, float* %cp.addr.052, i64 %idx.ext42
  br i1 false, label %for.body18, label %if.end

if.end:                                           ; preds = %for.body18
  ret void
}
