; RUN: not --crash llc -march=amdgcn -mcpu=gfx1200 < %s 2>&1 | FileCheck --check-prefixes=GCN-ERR %s

; GCN-ERR: LLVM ERROR: s_prefetch_data only supports global or constant memory
define amdgpu_ps void @prefetch_data_sgpr_base_imm_len_local(ptr addrspace(3) inreg %ptr) {
entry:
  tail call void @llvm.amdgcn.s.prefetch.data.p3(ptr addrspace(3) %ptr, i32 31)
  ret void
}
