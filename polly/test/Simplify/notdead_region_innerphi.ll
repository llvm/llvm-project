; RUN: opt %loadNPMPolly "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Do not remove dependencies of a phi node within a region statement (%phi).
;
define void @func(i32 %n, ptr noalias nonnull %A, double %alpha) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      br label %region_entry


    region_entry:
      %region.cmp = fcmp ueq double %alpha, 0.0
      br i1 %region.cmp, label %region_true, label %region_exit

    region_true:
      br i1 true, label %region_verytrue, label %region_mostlytrue

    region_verytrue:
      br label %region_mostlytrue

    region_mostlytrue:
      %phi = phi double [%val, %region_true], [0.0, %region_verytrue]
      store double %phi, ptr %A
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
