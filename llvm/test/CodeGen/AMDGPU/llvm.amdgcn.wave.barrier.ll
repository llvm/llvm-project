; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_wave_barrier:
; GCN-DAG: ; wave barrier
; GCN-NOT: s_barrier

define amdgpu_kernel void @test_wave_barrier() nounwind {
entry:
  call void @llvm.amdgcn.wave.barrier() convergent nounwind
  ret void
}

; Check for verifier error from interpreting wave_barrier as a control
; flow barrier.

; GCN-LABEL: {{^}}test_wave_barrier_is_not_isBarrier:
; GCN-DAG: ; wave barrier
; GCN-NOT: s_barrier
define amdgpu_kernel void @test_wave_barrier_is_not_isBarrier() nounwind {
entry:
  call void @llvm.amdgcn.wave.barrier() convergent nounwind
  unreachable
}

declare void @llvm.amdgcn.wave.barrier() convergent nounwind
