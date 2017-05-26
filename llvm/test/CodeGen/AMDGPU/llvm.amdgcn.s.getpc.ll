; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.s.getpc() #0

; GCN-LABEL: {{^}}test_s_getpc:
; GCN: s_load_dwordx2
; GCN-DAG: s_getpc_b64 s{{\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @test_s_getpc(i64 addrspace(1)* %out) #0 {
  %tmp = call i64 @llvm.amdgcn.s.getpc() #1
  store volatile i64 %tmp, i64 addrspace(1)* %out, align 8
  ret void
}

attributes #0 = { nounwind readnone speculatable }
