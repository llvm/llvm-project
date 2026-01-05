; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_ps void @prefetch_data_sgpr_base_imm_len_local(ptr addrspace(3) inreg %ptr) {
entry:
  ; CHECK: llvm.amdgcn.s.prefetch.data only supports global or constant memory
  tail call void @llvm.amdgcn.s.prefetch.data.p3(ptr addrspace(3) %ptr, i32 31)
  ret void
}
