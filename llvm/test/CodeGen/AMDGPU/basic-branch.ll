; RUN: llc -O0 -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCNNOOPT -check-prefix=GCN %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-spill-sgpr-to-smem=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope  -check-prefix=GCNNOOPT -check-prefix=GCN %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCNOPT -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCNOPT -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_branch:
; GCNNOOPT: v_writelane_b32
; GCNNOOPT: v_writelane_b32
; GCN: s_cbranch_scc1 [[END:BB[0-9]+_[0-9]+]]

; GCNNOOPT: v_readlane_b32
; GCNNOOPT: v_readlane_b32
; GCN: buffer_store_dword
; GCNNOOPT: s_endpgm

; GCN: {{^}}[[END]]:
; GCN: s_endpgm
define amdgpu_kernel void @test_branch(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %val) #0 {
  %cmp = icmp ne i32 %val, 0
  br i1 %cmp, label %store, label %end

store:
  store i32 222, i32 addrspace(1)* %out
  ret void

end:
  ret void
}

; GCN-LABEL: {{^}}test_brcc_i1:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCNNOOPT: s_and_b32 s{{[0-9]+}}, 1, [[VAL]]
; GCNOPT: s_and_b32 s{{[0-9]+}}, [[VAL]], 1
; GCN: s_cmp_eq_u32
; GCN: s_cbranch_scc1 [[END:BB[0-9]+_[0-9]+]]

; GCN: buffer_store_dword

; GCN: {{^}}[[END]]:
; GCN: s_endpgm
define amdgpu_kernel void @test_brcc_i1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i1 %val) #0 {
  %cmp0 = icmp ne i1 %val, 0
  br i1 %cmp0, label %store, label %end

store:
  store i32 222, i32 addrspace(1)* %out
  ret void

end:
  ret void
}

attributes #0 = { nounwind }
