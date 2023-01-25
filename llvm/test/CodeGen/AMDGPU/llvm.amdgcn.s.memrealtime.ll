; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.s.memrealtime

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
