; RUN: opt %loadNPMPolly '-passes=polly-import-jscop,print<polly-simplify>' -polly-import-jscop-postfix=transformed -disable-output < %s | FileCheck %s -match-full-lines
;
; Remove redundant store (a store that writes the same value already
; at the destination) in a region.
;
define void @redundant_region(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %region_entry, label %exit


    region_entry:
      %val = load double, ptr %A
      %cmp = fcmp oeq double %val, 0.0
      br i1 %cmp, label %region_true, label %region_exit

    region_true:
      br label %region_exit

    region_exit:
      br label %body

    body:
      store double %val, ptr %A
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
; CHECK:     Redundant writes removed: 2
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
