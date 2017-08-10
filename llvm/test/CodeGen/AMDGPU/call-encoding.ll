; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -filetype=obj -verify-machineinstrs < %s | llvm-objdump -triple amdgcn--amdhsa -mcpu=fiji -d - | FileCheck -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -verify-machineinstrs < %s | llvm-objdump -triple amdgcn--amdhsa -mcpu=gfx900 -d - | FileCheck -check-prefixes=GCN,GFX9 %s
; XUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -filetype=obj -verify-machineinstrs < %s | llvm-objdump -triple amdgcn--amdhsa -mcpu=hawaii -d - | FileCheck -check-prefixes=GCN,CI %s

; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @void_func_void() #1 {
  ret void
}

; GCN: s_getpc_b64
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_void_func_void() {
  call void @void_func_void()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline }
