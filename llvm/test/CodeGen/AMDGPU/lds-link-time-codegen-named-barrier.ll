; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-enable-object-linking < %s | FileCheck %s

; Verify object linking codegen for named barriers on GFX1250:
; 1. Barrier instructions use M0-based forms with relocation references
; 2. group_segment_fixed_size = 0 (linker patches it)
; 3. Named barrier is emitted as an SHN_AMDGPU_LDS symbol (.amdgpu_lds)

@bar = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison

; CHECK-LABEL: kernel:
; CHECK: s_lshr_b32 s{{[0-9]+}}, __amdgpu_named_barrier.bar{{[^ @]*}}@abs32@lo, 4
; CHECK: s_barrier_signal m0
; CHECK: s_barrier_join m0
; CHECK: s_barrier_wait 1

; KD: group_segment_fixed_size = 0 (linker will patch).
; CHECK:       .amdhsa_group_segment_fixed_size 0

; LDS symbol declaration
; CHECK:      .amdgpu_lds __amdgpu_named_barrier.bar{{[^ ,]*}}, 32, 4

define amdgpu_kernel void @kernel() {
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 3)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @helper()
  ret void
}

declare void @helper()
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32) #0
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3)) #0
declare void @llvm.amdgcn.s.barrier.wait(i16) #0

attributes #0 = { convergent nounwind }
