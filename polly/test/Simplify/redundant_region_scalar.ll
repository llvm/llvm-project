; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
;
; Remove redundant store (a store that writes the same value already
; at the destination) in a region.
;
define void @redundant_region_scalar(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit


    bodyA:
      %val1 = load double, double* %A
      br label %region_entry

    region_entry:
      %val2 = load double, double* %A
      %cmp = fcmp oeq double %val1, 0.0
      br i1 %cmp, label %region_true, label %region_exit

    region_true:
      br label %region_exit

    region_exit:
      br label %bodyB

    bodyB:
      store double %val2, double* %A
      br label %inc


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Redundant writes removed: 3
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
