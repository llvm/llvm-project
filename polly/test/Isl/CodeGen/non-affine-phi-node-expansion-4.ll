; RUN: opt %loadPolly -polly-codegen \
; RUN:     -S < %s | FileCheck %s

define void @foo(float* %A, i1 %cond0, i1 %cond1) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %backedge]
  %val0 = fadd float 1.0, 2.0
  %val1 = fadd float 1.0, 2.0
  br i1 %cond0, label %branch1, label %backedge

; CHECK-LABEL: polly.stmt.loop:
; CHECK-NEXT:    %p_val0 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT:    %p_val1 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT:    br i1

; The interesting instruction here is %val2, which does not dominate the exit of
; the non-affine region. Care needs to be taken when code-generating this write.
; Specifically, at some point we modeled this scalar write, which we tried to
; code generate in the exit block of the non-affine region.
branch1:
  %val2 = fadd float 1.0, 2.0
  br i1 %cond1, label %branch2, label %backedge

; CHECK-LABEL: polly.stmt.branch1:
; CHECK-NEXT:    %p_val2 = fadd float 1.000000e+00, 2.000000e+00
; CHECK-NEXT:    br i1

branch2:
  br label %backedge

; CHECK-LABEL: polly.stmt.branch2:
; CHECK-NEXT:    br label

; CHECK-LABEL: polly.stmt.backedge.exit:
; CHECK:         %polly.merge = phi float [ %p_val0, %polly.stmt.loop ], [ %p_val1, %polly.stmt.branch1 ], [ %p_val2, %polly.stmt.branch2 ]

backedge:
  %merge = phi float [%val0, %loop], [%val1, %branch1], [%val2, %branch2]
  %indvar.next = add i64 %indvar, 1
  store float %merge, float* %A
  %cmp = icmp sle i64 %indvar.next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
