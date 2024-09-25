; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [dim] -> {  : dim > 0 }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [dim] -> {  : dim < 0 }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @horner_bezier_curve(ptr %cp, i32 %dim) #0 {
entry:
  br label %for.body18.lr.ph

for.body18.lr.ph:                                 ; preds = %entry
  br label %for.body18

for.body18:                                       ; preds = %for.body18, %for.body18.lr.ph
  %cp.addr.052 = phi ptr [ %cp, %for.body18.lr.ph ], [ %add.ptr43, %for.body18 ]
  %0 = load float, ptr %cp.addr.052, align 4
  store float %0, ptr %cp.addr.052, align 4
  %idx.ext42 = zext i32 %dim to i64
  %add.ptr43 = getelementptr inbounds float, ptr %cp.addr.052, i64 %idx.ext42
  br i1 false, label %for.body18, label %if.end

if.end:                                           ; preds = %for.body18
  ret void
}
