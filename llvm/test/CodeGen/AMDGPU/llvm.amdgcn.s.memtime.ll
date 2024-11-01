; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck --check-prefixes=SIVI,GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=SIVI,GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.s.memtime() #0

; GCN-LABEL: {{^}}test_s_memtime:
; GCN-DAG: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN-DAG: s_load_dwordx2
; GCN: lgkmcnt
; GCN: {{buffer|global}}_store_dwordx2
; SIVI-NOT: lgkmcnt
; GCN: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN: {{buffer|global}}_store_dwordx2
define amdgpu_kernel void @test_s_memtime(ptr addrspace(1) %out) #0 {
  %cycle0 = call i64 @llvm.amdgcn.s.memtime()
  store volatile i64 %cycle0, ptr addrspace(1) %out

  %cycle1 = call i64 @llvm.amdgcn.s.memtime()
  store volatile i64 %cycle1, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
