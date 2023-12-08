; RUN: opt %loadPolly -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Do not remove redundant stores in the middle of region statements.
; The store in region_true could be removed, but in practice we do try to
; determine the relative ordering of block in region statements.
;
; for (int j = 0; j < n; j += 1) {
;   double val = A[0];
;   if (val == 0.0)
;     A[0] = val;
;   else
;     A[0] = 0.0;
; }
;
define void @notredundant_region(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %region_entry, label %exit


    region_entry:
      %val = load double, ptr %A
      %cmp = fcmp oeq double %val, 0.0
      br i1 %cmp, label %region_true, label %region_false

    region_true:
      store double %val, ptr %A
      br label %region_exit

    region_false:
      store double 0.0, ptr %A
      br label %region_exit

    region_exit:
      br label %inc


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: SCoP could not be simplified
