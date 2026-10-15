; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @prefetch_data_sgpr_base_imm_len_local(ptr addrspace(3) inreg %ptr) {
entry:
  ; CHECK: llvm.amdgcn.s.prefetch.data only supports global or constant memory
  tail call void @llvm.amdgcn.s.prefetch.data.p3(ptr addrspace(3) %ptr, i32 31)
  ret void
}

define amdgpu_ps void @global_prefetch_reserved_th(ptr addrspace(1) %ptr) {
  ; CHECK: invalid cache policy for prefetch
  ; CHECK-NEXT: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %ptr, i32 7)
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %ptr, i32 7)
  ret void
}

define amdgpu_ps void @flat_prefetch_extra_cpol_bits(ptr %ptr) {
  ; CHECK: invalid cache policy for prefetch
  ; CHECK-NEXT: call void @llvm.amdgcn.flat.prefetch(ptr %ptr, i32 2048)
  call void @llvm.amdgcn.flat.prefetch(ptr %ptr, i32 2048)
  ret void
}
