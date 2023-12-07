; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; Test with zero frame
; GCN-LABEL: {{^}}func1
; GCN: v_mov_b32_e32 v0, s30
; GCN: v_mov_b32_e32 v1, s31
; GCN: s_setpc_b64 s[30:31]
define ptr @func1() nounwind {
entry:
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

; Test with non-zero frame
; GCN-LABEL: {{^}}func2
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, 0
; GCN: s_setpc_b64 s[30:31]
define ptr @func2() nounwind {
entry:
  %0 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %0
}

; Test with amdgpu_kernel
; GCN-LABEL: {{^}}func3
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, {{v0|0}}
define amdgpu_kernel void @func3(ptr %out) nounwind {
entry:
  %tmp = tail call ptr @llvm.returnaddress(i32 0)
  store ptr %tmp, ptr %out, align 4
  ret void
}

; Test with use outside the entry-block
; GCN-LABEL: {{^}}func4
; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v1, {{v0|0}}
define amdgpu_kernel void @func4(ptr %out, i32 %val) nounwind {
entry:
  %cmp = icmp ne i32 %val, 0
  br i1 %cmp, label %store, label %exit

store:
  %tmp = tail call ptr @llvm.returnaddress(i32 1)
  store ptr %tmp, ptr %out, align 4
  ret void

exit:
  ret void
}

; Test ending in unreachable
; GCN-LABEL: {{^}}func5
; GCN: v_mov_b32_e32 v0, 0
define void @func5() nounwind {
entry:
  %tmp = tail call ptr @llvm.returnaddress(i32 2)
  store volatile i32 0, ptr addrspace(3) undef, align 4
  unreachable
}

declare void @callee()

; GCN-LABEL: {{^}}multi_use:
; GCN-DAG: v_mov_b32_e32 v[[LO:4[0-9]+]], s30
; GCN-DAG: v_mov_b32_e32 v[[HI:4[0-9]+]], s31
; GCN: global_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
; GCN: s_swappc_b64
; GCN: global_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define void @multi_use() nounwind {
entry:
  %ret0 = tail call ptr @llvm.returnaddress(i32 0)
  store volatile ptr %ret0, ptr addrspace(1) undef
  call void @callee()
  %ret1 = tail call ptr @llvm.returnaddress(i32 0)
  store volatile ptr %ret1, ptr addrspace(1) undef
  ret void
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
