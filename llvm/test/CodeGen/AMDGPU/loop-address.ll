;RUN: llc < %s -march=r600 -mcpu=redwood < %s | FileCheck %s

;CHECK: ALU_PUSH
;CHECK: LOOP_START_DX10 @11
;CHECK: LOOP_BREAK @10
;CHECK: POP @10

define amdgpu_kernel void @loop_ge(ptr addrspace(1) nocapture %out, i32 %iterations) #0 {
entry:
  %cmp5 = icmp sgt i32 %iterations, 0
  br i1 %cmp5, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.07.in = phi i32 [ %i.07, %for.body ], [ %iterations, %entry ]
  %ai.06 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %i.07 = add nsw i32 %i.07.in, -1
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %ai.06
  store i32 %i.07, ptr addrspace(1) %arrayidx, align 4
  %add = add nsw i32 %ai.06, 1
  %exitcond = icmp eq i32 %add, %iterations
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }

!opencl.kernels = !{!0, !1, !2, !3}

!0 = !{ptr @loop_ge}
!1 = !{null}
!2 = !{null}
!3 = !{null}
