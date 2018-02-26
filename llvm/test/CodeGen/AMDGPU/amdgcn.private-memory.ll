; RUN: llc -mattr=+promote-alloca -verify-machineinstrs -march=amdgcn < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-PROMOTE %s
; RUN: llc -mattr=+promote-alloca,-flat-for-global -verify-machineinstrs -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-PROMOTE -check-prefix=HSA %s
; RUN: llc -mattr=-promote-alloca -verify-machineinstrs -march=amdgcn < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-ALLOCA %s
; RUN: llc -mattr=-promote-alloca,-flat-for-global -verify-machineinstrs -mtriple=amdgcn-amdhsa -mcpu=kaveri < %s | FileCheck  -check-prefix=GCN -check-prefix=GCN-ALLOCA -check-prefix=HSA %s
; RUN: llc -mattr=+promote-alloca -verify-machineinstrs -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-PROMOTE %s
; RUN: llc -mattr=-promote-alloca -verify-machineinstrs -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN  -check-prefix=GCN-ALLOCA %s


declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone


; Make sure we don't overwrite workitem information with private memory

; GCN-LABEL: {{^}}work_item_info:
; GCN-NOT: v0
; GCN: v_add_{{[iu]}}32_e32 [[RESULT:v[0-9]+]], vcc, v0, v{{[0-9]+}}
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @work_item_info(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = alloca [2 x i32]
  %1 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 0
  %2 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 1
  store i32 0, i32* %1
  store i32 1, i32* %2
  %3 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 %in
  %4 = load i32, i32* %3
  %5 = call i32 @llvm.amdgcn.workitem.id.x()
  %6 = add i32 %4, %5
  store i32 %6, i32 addrspace(1)* %out
  ret void
}
