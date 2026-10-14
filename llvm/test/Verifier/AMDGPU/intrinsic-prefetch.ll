; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @prefetch_data_sgpr_base_imm_len_local(ptr addrspace(3) inreg %ptr) {
entry:
  ; CHECK: llvm.amdgcn.s.prefetch.data only supports global or constant memory
  tail call void @llvm.amdgcn.s.prefetch.data.p3(ptr addrspace(3) %ptr, i32 31)
  ret void
}

define amdgpu_ps void @prefetch_data_len_too_large(ptr addrspace(1) inreg %ptr) {
  ; CHECK: prefetch length must be in the range [0, 31]
  ; CHECK-NEXT: call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) %ptr, i32 32)
  call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) %ptr, i32 32)
  ret void
}

define amdgpu_ps void @prefetch_inst_len_too_large(ptr inreg %ptr) {
  ; CHECK: prefetch length must be in the range [0, 31]
  ; CHECK-NEXT: call void @llvm.amdgcn.s.prefetch.inst.p0(ptr %ptr, i32 255)
  call void @llvm.amdgcn.s.prefetch.inst.p0(ptr %ptr, i32 255)
  ret void
}

define amdgpu_ps void @buffer_prefetch_data_len_too_large(ptr addrspace(8) inreg %rsrc) {
  ; CHECK: prefetch length must be in the range [0, 31]
  ; CHECK-NEXT: call void @llvm.amdgcn.s.buffer.prefetch.data(ptr addrspace(8) %rsrc, i32 0, i32 64)
  call void @llvm.amdgcn.s.buffer.prefetch.data(ptr addrspace(8) %rsrc, i32 0, i32 64)
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
