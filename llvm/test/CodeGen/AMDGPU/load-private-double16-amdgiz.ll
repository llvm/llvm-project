; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgiz -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"

; GCN-LABEL: @test_unaligned_load
; GCN: buffer_load_dword
; GCN-NOT: flat_load_dword
define amdgpu_kernel void @test_unaligned_load(<16 x double> addrspace(1)* %results, i32 %i) {
entry:
  %a = inttoptr i32 %i to <16 x double> addrspace(5)*
  %v = load <16 x double>, <16 x double> addrspace(5)* %a, align 8 
  store <16 x double> %v, <16 x double> addrspace(1)* %results, align 128
  ret void
}

; GCN-LABEL: @test_unaligned_store
; GCN: buffer_store_dword
; GCN-NOT: flat_store_dword
define amdgpu_kernel void @test_unaligned_store(<16 x double> %v, i32 %i) {
entry:
  %a = inttoptr i32 %i to <16 x double> addrspace(5)*
  store <16 x double> %v, <16 x double> addrspace(5)* %a, align 8
  ret void
}
