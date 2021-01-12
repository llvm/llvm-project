; RUN: llc -march=amdgcn -verify-machineinstrs -amdgpu-scalar-ir-passes=false < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -amdgpu-scalar-ir-passes=false < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs -amdgpu-scalar-ir-passes=false < %s | FileCheck -check-prefix=GCN %s

; These tests are split out from umed3.ll and smed3.ll and use the
; -amdgpu-scalar-ir-passes=false flag, because InstSimplify would constant
; fold these functions otherwise.

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_constant_order_i32:
; GCN: v_max_u32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
; GCN: v_min_u32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
define amdgpu_kernel void @v_test_umed3_r_i_i_constant_order_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp ugt i32 %a, 17
  %i0 = select i1 %icmp0, i32 %a, i32 17

  %icmp1 = icmp ult i32 %i0, 12
  %i1 = select i1 %icmp1, i32 %i0, i32 12

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_constant_order_i32:
; GCN: v_max_i32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
; GCN: v_min_i32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
define amdgpu_kernel void @v_test_smed3_r_i_i_constant_order_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp sgt i32 %a, 17
  %i0 = select i1 %icmp0, i32 %a, i32 17

  %icmp1 = icmp slt i32 %i0, 12
  %i1 = select i1 %icmp1, i32 %i0, i32 12

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

