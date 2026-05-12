; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-enable-object-linking < %s | FileCheck %s --implicit-check-not=.amdgpu_num_agpr
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --sections - | FileCheck %s --check-prefix=ELF

; Verify object linking codegen for named barriers on GFX1250:
; 1. Barrier instructions use M0-based forms with relocation references
; 2. .amdgpu.info section records the barrier as an LDS use edge
; 3. group_segment_fixed_size = 0 (linker patches it)
; 4. Named barrier is emitted as an SHN_AMDGPU_LDS symbol (.amdgpu_lds)

@bar = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison

; CHECK-LABEL: kernel:
; CHECK: s_lshr_b32 s{{[0-9]+}}, __amdgpu_named_barrier.bar{{[^ @]*}}@abs32@lo, 4
; CHECK: s_barrier_join m0
; CHECK: s_barrier_signal m0
; CHECK: s_barrier_wait 1

; CHECK:       .amdhsa_group_segment_fixed_size 0

; CHECK:      .amdgpu_info kernel
; CHECK:        .amdgpu_flags {{[0-9]+}}
; CHECK:        .amdgpu_num_sgpr {{[0-9]+}}
; CHECK:        .amdgpu_num_vgpr {{[0-9]+}}
; CHECK:        .amdgpu_private_segment_size {{[0-9]+}}
; CHECK:        .amdgpu_use __amdgpu_named_barrier.bar{{[^ ,]*}}
; CHECK:        .amdgpu_call helper
; CHECK:      .end_amdgpu_info

; CHECK:      .amdgpu_lds __amdgpu_named_barrier.bar{{[^ ,]*}}, 32, 4

; ELF:      Section {
; ELF:        Name: .amdgpu.info
; ELF:        Type: SHT_PROGBITS
; ELF:        Flags [
; ELF:          SHF_EXCLUDE

; ELF-DAG: R_AMDGPU_ABS64 kernel
; ELF-DAG: R_AMDGPU_ABS64 __amdgpu_named_barrier.bar{{[^ ]*}}
; ELF-DAG: R_AMDGPU_ABS64 helper

define amdgpu_kernel void @kernel() {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 3)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @helper()
  ret void
}

declare void @helper()
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3)) #0
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32) #0
declare void @llvm.amdgcn.s.barrier.wait(i16) #0

attributes #0 = { convergent nounwind }
