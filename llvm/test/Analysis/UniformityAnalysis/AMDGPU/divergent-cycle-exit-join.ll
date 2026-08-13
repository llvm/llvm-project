; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Test for a bug where the uniformity analysis fails to mark a cycle exit PHI
; as divergent when its incoming values are constants from multiple cycle
; predecessors.
;
; The divergent branch in body (br i1 %c) has one target outside the loop
; (exit) and one on the back-edge (loop). The exit block is also reachable
; from the loop header via a uniform branch. Different lanes exit from
; different predecessors, making the exit PHI join-divergent.
;
; BUG: The uniformity analysis incorrectly classifies %exit.phi as uniform.

; CHECK-LABEL: UniformityInfo for function 'constant_phi_at_cycle_exit':
; CHECK: CYCLES WITH DIVERGENT EXIT:
; CHECK-NOT: DIVERGENT:{{.*}}%exit.phi
; CHECK: %exit.phi = phi i32 [ 1, %body ], [ 0, %loop ]

declare i32 @llvm.amdgcn.workitem.id.x() #0

define amdgpu_kernel void @constant_phi_at_cycle_exit(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %cond.exit = icmp slt i32 %ctr, 1
  br i1 %cond.exit, label %body, label %exit

body:
  %c = icmp eq i32 %tid, 0
  br i1 %c, label %exit, label %loop

exit:
  %exit.phi = phi i32 [ 1, %body ], [ 0, %loop ]
  store i32 %exit.phi, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone speculatable }
