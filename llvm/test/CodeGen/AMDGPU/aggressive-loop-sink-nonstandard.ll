; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 --aggressively-sink-insts-to-avoid-spills=1  < %s | FileCheck -check-prefix=SUNK %s

; Check that various edge cases do not crash the compiler

; Multiple uses of sunk valu, chain of sink candidates

define half @global_agent_atomic_fmin_ret_f16__amdgpu_no_fine_grained_memory(ptr addrspace(1) %ptr, half %val) {
; SUNK-LABEL: global_agent_atomic_fmin_ret_f16__amdgpu_no_fine_grained_memory:
  %result = atomicrmw fmin ptr addrspace(1) %ptr, half %val syncscope("agent") seq_cst
  ret half %result
}

; Sink candidates with multiple defs

define void @memmove_p5_p5(ptr addrspace(5) align 1 %dst, ptr addrspace(5) align 1 readonly %src, i64 %sz) {
; SUNK-LABEL: memmove_p5_p5:
entry:
  tail call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}
