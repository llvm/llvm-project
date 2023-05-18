; RUN: llc -mattr=+promote-alloca -verify-machineinstrs -mtriple=amdgcn < %s | FileCheck --check-prefixes=GCN,GCN-PROMOTE %s
; RUN: llc -mattr=+promote-alloca,-flat-for-global -verify-machineinstrs -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck --check-prefixes=GCN,GCN-PROMOTE %s
; RUN: llc -mattr=-promote-alloca -verify-machineinstrs -mtriple=amdgcn < %s | FileCheck --check-prefixes=GCN,GCN-ALLOCA %s
; RUN: llc -mattr=-promote-alloca,-flat-for-global -verify-machineinstrs -mtriple=amdgcn-amdhsa -mcpu=kaveri < %s | FileCheck  --check-prefixes=GCN,GCN-ALLOCA %s
; RUN: llc -mattr=+promote-alloca -verify-machineinstrs -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN,GCN-PROMOTE %s
; RUN: llc -mattr=-promote-alloca -verify-machineinstrs -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN,GCN-ALLOCA %s


declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone


; Make sure we don't overwrite workitem information with private memory

; GCN-LABEL: {{^}}work_item_info:
; GCN-NOT: v0
; GCN: s_load_dword [[IN:s[0-9]+]]
; GCN-NOT: v0

; GCN-ALLOCA: v_add_{{[iu]}}32_e32 [[RESULT:v[0-9]+]], vcc, v{{[0-9]+}}, v0

; GCN-PROMOTE: s_cmp_eq_u32 [[IN]], 1
; GCN-PROMOTE-NEXT: s_cselect_b64 vcc, -1, 0
; GCN-PROMOTE-NEXT: v_addc_u32_e32 [[RESULT:v[0-9]+]], vcc, 0, v0, vcc

; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @work_item_info(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = alloca [2 x i32], addrspace(5)
  %1 = getelementptr [2 x i32], ptr addrspace(5) %0, i32 0, i32 1
  store i32 0, ptr addrspace(5) %0
  store i32 1, ptr addrspace(5) %1
  %2 = getelementptr [2 x i32], ptr addrspace(5) %0, i32 0, i32 %in
  %3 = load i32, ptr addrspace(5) %2
  %4 = call i32 @llvm.amdgcn.workitem.id.x()
  %5 = add i32 %3, %4
  store i32 %5, ptr addrspace(1) %out
  ret void
}
