; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Remove a dead instruction
; (an instruction whose result is not used anywhere)
;
; for (int j = 0; j < n; j += 1) {
;   double val = 21.0 + 21.0;
;   A[0] = 42.0;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      store double 42.0, ptr %A
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
; CHECK:     Dead instructions removed: 1
; CHECK: }
