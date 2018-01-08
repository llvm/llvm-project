; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-allow-nonaffine-loops -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; Do not remove the store in region_entry. It can be executed multiple times
; due to being part of a non-affine loop.
;
define void @notredundant_region_loop(i32 %n, double* noalias nonnull %A) {
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
      store double %val, double* %A
      %sqr = mul i32 %j, %j
      %cmp = icmp eq i32 %sqr, 42
      br i1 %cmp, label %region_true, label %region_exit

    region_true:
      store double 0.0, double* %A
      br label %region_entry

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
