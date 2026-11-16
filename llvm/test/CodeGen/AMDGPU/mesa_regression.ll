; RUN: llc -O2 -mtriple=amdgpu8.03--amdhsa < %s | FileCheck %s

; CHECK-LABEL: %entry
; CHECK: flat_load_dwordx4

define amdgpu_kernel void @store_global(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture readonly %in) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds <16 x double>, ptr addrspace(1) %in, i32 %tid
  %tmp = load <16 x double>, ptr addrspace(1) %in.tid
  store <16 x double> %tmp, ptr addrspace(1) %out
  ret void
}
