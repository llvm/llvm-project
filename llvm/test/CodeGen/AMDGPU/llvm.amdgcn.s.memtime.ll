; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

declare i64 @llvm.amdgcn.s.memtime() #0

; GCN-LABEL: {{^}}test_s_memtime:
; GCN-DAG: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN-DAG: s_load_dwordx2
; GCN: lgkmcnt
; GCN: buffer_store_dwordx2
; GCN-NOT: lgkmcnt
; GCN: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dwordx2
define void @test_s_memtime(i64 addrspace(1)* %out) #0 {
  %cycle0 = call i64 @llvm.amdgcn.s.memtime()
  store volatile i64 %cycle0, i64 addrspace(1)* %out

  %cycle1 = call i64 @llvm.amdgcn.s.memtime()
  store volatile i64 %cycle1, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
