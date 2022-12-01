; RUN: opt -S -passes=indvars < %s | FileCheck %s

; This is not an IndVarSimplify bug, but the original symptom
; manifested as one.

define i32 @foo(i32 %a, i32 %b, i32 %c, i32* %sink) {
; CHECK-LABEL: @foo(
; CHECK:       for.end:
; CHECK-NEXT:    [[SHR:%.*]] = ashr i32 %neg3, -1
; CHECK-NEXT:    [[SUB:%.*]] = sub nsw i32 0, [[SHR]]
; CHECK-NEXT:    [[SHR1:%.*]] = ashr i32 [[SUB]], [[B:%.*]]
; CHECK-NEXT:    [[NEG:%.*]] = xor i32 [[SHR1]], -1
; CHECK-NEXT:    store i32 [[NEG]], i32* %sink
;
entry:
  %tobool2 = icmp eq i32 %a, 0
  br i1 %tobool2, label %exit, label %preheader

preheader:
  %neg3 = phi i32 [ %c, %entry ], [ %neg, %for.end ]
  br label %for

for:
  %p = phi i32 [ %dec, %for ], [ 1, %preheader ]
  %cmp = icmp sgt i32 %p, -1
  %dec = add nsw i32 %p, -1
  br i1 %cmp, label %for, label %for.end

for.end:
  %shr = ashr i32 %neg3, %p
  %sub = sub nsw i32 0, %shr
  %shr1 = ashr i32 %sub, %b
  %neg = xor i32 %shr1, -1
  store i32 %neg, i32* %sink
  br i1 false, label %exit, label %preheader

exit:
  ret i32 0
}
