; Test if "polly-debug" flag enables debug prints from different parts of polly
; RUN: opt %loadNPMPolly -O3 -polly -polly-debug --disable-output < %s 2>&1 | FileCheck %s
;
; REQUIRES: asserts

; void callee(int n, double A[], int i) {
;   for (int j = 0; j < n; j += 1)
;     A[i+j] = 42.0;
; }
;
; void caller(int n, double A[]) {
;   for (int i = 0; i < n; i += 1)
;     callee(n, A, i);
; }


%unrelated_type = type { i32 }

define internal void @callee(i32 %n, ptr noalias nonnull %A, i32 %i) #0 {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %idx = add i32 %i, %j
      %arrayidx = getelementptr inbounds double, ptr %A, i32 %idx
      store double 42.0, ptr %arrayidx
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


define void @caller(i32 %n, ptr noalias nonnull %A) #0 {
entry:
  br label %for

for:
  %i = phi i32 [0, %entry], [%j.inc, %inc]
  %i.cmp = icmp slt i32 %i, %n
  br i1 %i.cmp, label %body, label %exit

    body:
      call void @callee(i32 %n, ptr %A, i32 %i)
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %i, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


declare void @unrelated_decl()


attributes #0 = { noinline }

!llvm.ident = !{!8}
!8 = !{!"xyxxy"}

; CHECK: Checking region: entry => <Function Return>
; CHECK: Removing statements that are never executed...
; CHECK: Final Scop:
; CHECK: Forwarding operand trees...
; CHECK: Final Scop:
; CHECK: Collapsing scalars to unused array elements...
; CHECK: Final Scop:
