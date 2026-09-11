; RUN: llc -global-isel=0 -mtriple=amdgpu8.03-amd-amdhsa -filetype=obj < %s | llvm-objdump --triple=amdgpu8.03--amdhsa -d - | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=0 -mtriple=amdgpu9.00-amd-amdhsa -filetype=obj < %s | llvm-objdump --triple=amdgpu9.00--amdhsa -d - | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgpu8.03-amd-amdhsa -filetype=obj < %s | llvm-objdump --triple=amdgpu8.03--amdhsa -d - | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgpu9.00-amd-amdhsa -filetype=obj < %s | llvm-objdump --triple=amdgpu9.00--amdhsa -d - | FileCheck --check-prefix=GCN %s
; XUN: llc -mtriple=amdgpu7.01-amd-amdhsa -filetype=obj < %s | llvm-objdump --triple=amdgpu7.01--amdhsa -d - | FileCheck --check-prefixes=GCN,CI %s

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
