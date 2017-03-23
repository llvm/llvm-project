; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; PRED_SET* instructions must be tied to any instruction that uses their
; result.  This tests that there are no instructions between the PRED_SET*
; and the PREDICATE_BREAK in this loop.

; CHECK: {{^}}loop_ge:
; CHECK: LOOP_START_DX10
; CHECK: ALU_PUSH_BEFORE
; CHECK-NEXT: JUMP
; CHECK-NEXT: LOOP_BREAK
define amdgpu_kernel void @loop_ge(i32 addrspace(1)* nocapture %out, i32 %iterations) nounwind {
entry:
  %cmp5 = icmp sgt i32 %iterations, 0
  br i1 %cmp5, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.07.in = phi i32 [ %i.07, %for.body ], [ %iterations, %entry ]
  %ai.06 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %i.07 = add nsw i32 %i.07.in, -1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %ai.06
  store i32 %i.07, i32 addrspace(1)* %arrayidx, align 4
  %add = add nsw i32 %ai.06, 1
  %exitcond = icmp eq i32 %add, %iterations
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
