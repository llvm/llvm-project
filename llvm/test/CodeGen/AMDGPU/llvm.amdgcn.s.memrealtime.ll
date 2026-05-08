; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=GCN %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 < %s 2>&1 | FileCheck -check-prefix=ERR-SDAG %s
; RUN: not llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn -mcpu=gfx1100 < %s 2>&1 | FileCheck -check-prefix=ERR-GISEL %s

; ERR-SDAG: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.s.memrealtime
; ERR-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:sreg_64(s64) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.s.memrealtime)

declare i64 @llvm.amdgcn.s.memrealtime() #0

; GCN-LABEL: {{^}}test_s_memrealtime:
; GCN-DAG: s_memrealtime s{{\[[0-9]+:[0-9]+\]}}
; GCN-DAG: s_load_dwordx2
; GCN: lgkmcnt
; GCN: _store_dwordx2
; GCN-NOT: lgkmcnt
; GCN: s_memrealtime s{{\[[0-9]+:[0-9]+\]}}
; GCN: _store_dwordx2
define amdgpu_kernel void @test_s_memrealtime(ptr addrspace(1) %out) #0 {
  %cycle0 = call i64 @llvm.amdgcn.s.memrealtime()
  store volatile i64 %cycle0, ptr addrspace(1) %out

  %cycle1 = call i64 @llvm.amdgcn.s.memrealtime()
  store volatile i64 %cycle1, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
