; RUN: opt %loadNPMPolly -polly-optree-normalize-phi=true '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
;
; Contains a self-referencing PHINode that would require a
; transitive closure to handle.
;
; for (int j = 0; j < n; j += 1) {
;   double phi = 0.0;
;   for (int i = 0; i < m; i += 1)
;     phi = phi;
;   A[j] = phi;
; }
;
define void @func(i32 %n, i32 %m, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %for.preheader, label %exit

  for.preheader:
    br label %for.inner

  for.inner:
    %i = phi i32 [0, %for.preheader], [%i.inc, %for.inner]
    %phi = phi double [0.0, %for.preheader], [%phi, %for.inner]
    %i.inc = add nuw nsw i32 %i, 1
    %i.cmp = icmp slt i32 %i.inc, %m
    br i1 %i.cmp, label %for.inner, label %for.exit

  for.exit:
    %A_idx = getelementptr inbounds double, ptr %A, i32 %j
    store double %phi, ptr %A_idx
    br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: ForwardOpTree executed, but did not modify anything
