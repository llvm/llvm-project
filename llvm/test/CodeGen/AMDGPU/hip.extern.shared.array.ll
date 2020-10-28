; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck %s

@lds0 = addrspace(3) global [512 x float] undef
@lds1 = addrspace(3) global [256 x float] undef
@lds2 = addrspace(3) global [4096 x float] undef
@lds3 = addrspace(3) global [67 x i8] undef

@dynamic_shared0 = external addrspace(3) global [0 x float]
@dynamic_shared1 = external addrspace(3) global [0 x double]
@dynamic_shared2 = external addrspace(3) global [0 x double], align 4
@dynamic_shared3 = external addrspace(3) global [0 x double], align 16

; CHECK-LABEL: {{^}}dynamic_shared_array_0:
; CHECK: v_add_u32_e32 v{{[0-9]+}}, 0x800, v{{[0-9]+}}
define amdgpu_kernel void @dynamic_shared_array_0(float addrspace(1)* %out) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %tid.x
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val0, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; CHECK-LABEL: {{^}}dynamic_shared_array_1:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0xc00
; CHECK: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_1(float addrspace(1)* %out, i32 %cond) {
entry:
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idx.0 = add nsw i32 %tid.x, 64
  %tmp = icmp eq i32 %cond, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  br label %endif

else:                                             ; preds = %entry
  %arrayidx1 = getelementptr inbounds [256 x float], [256 x float] addrspace(3)* @lds1, i32 0, i32 %idx.0
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  br label %endif

endif:                                            ; preds = %else, %if
  %val = phi float [ %val0, %if ], [ %val1, %else ]
  %arrayidx = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val, float addrspace(3)* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: {{^}}dynamic_shared_array_2:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0x4000
; CHECK: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_2(i32 %idx) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %vidx = add i32 %tid.x, %idx
  %arrayidx0 = getelementptr inbounds [4096 x float], [4096 x float] addrspace(3)* @lds2, i32 0, i32 %vidx
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val0, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; The offset to the dynamic shared memory array should be aligned on the type
; specified.
; CHECK-LABEL: {{^}}dynamic_shared_array_3:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0x44
; CHECK: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_3(i32 %idx) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %vidx = add i32 %tid.x, %idx
  %arrayidx0 = getelementptr inbounds [67 x i8], [67 x i8] addrspace(3)* @lds3, i32 0, i32 %vidx
  %val0 = load i8, i8 addrspace(3)* %arrayidx0, align 4
  %val1 = uitofp i8 %val0 to float
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; The offset to the dynamic shared memory array should be aligned on the
; maximal one.
; CHECK-LABEL: {{^}}dynamic_shared_array_4:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0x48
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 3, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_4(i32 %idx) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %vidx = add i32 %tid.x, %idx
  %arrayidx0 = getelementptr inbounds [67 x i8], [67 x i8] addrspace(3)* @lds3, i32 0, i32 %vidx
  %val0 = load i8, i8 addrspace(3)* %arrayidx0, align 4
  %val1 = uitofp i8 %val0 to float
  %val2 = uitofp i8 %val0 to double
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [0 x double], [0 x double] addrspace(3)* @dynamic_shared1, i32 0, i32 %tid.x
  store double %val2, double addrspace(3)* %arrayidx2, align 4
  ret void
}

; Honor the explicit alignment from the specified variable.
; CHECK-LABEL: {{^}}dynamic_shared_array_5:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0x44
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 3, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_5(i32 %idx) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %vidx = add i32 %tid.x, %idx
  %arrayidx0 = getelementptr inbounds [67 x i8], [67 x i8] addrspace(3)* @lds3, i32 0, i32 %vidx
  %val0 = load i8, i8 addrspace(3)* %arrayidx0, align 4
  %val1 = uitofp i8 %val0 to float
  %val2 = uitofp i8 %val0 to double
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [0 x double], [0 x double] addrspace(3)* @dynamic_shared2, i32 0, i32 %tid.x
  store double %val2, double addrspace(3)* %arrayidx2, align 4
  ret void
}

; Honor the explicit alignment from the specified variable.
; CHECK-LABEL: {{^}}dynamic_shared_array_6:
; CHECK: v_mov_b32_e32 [[DYNLDS:v[0-9]+]], 0x50
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 2, [[DYNLDS]]
; CHECK-DAG: v_lshl_add_u32 {{v[0-9]+}}, {{v[0-9]+}}, 3, [[DYNLDS]]
define amdgpu_kernel void @dynamic_shared_array_6(i32 %idx) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %vidx = add i32 %tid.x, %idx
  %arrayidx0 = getelementptr inbounds [67 x i8], [67 x i8] addrspace(3)* @lds3, i32 0, i32 %vidx
  %val0 = load i8, i8 addrspace(3)* %arrayidx0, align 4
  %val1 = uitofp i8 %val0 to float
  %val2 = uitofp i8 %val0 to double
  %arrayidx1 = getelementptr inbounds [0 x float], [0 x float] addrspace(3)* @dynamic_shared0, i32 0, i32 %tid.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [0 x double], [0 x double] addrspace(3)* @dynamic_shared3, i32 0, i32 %tid.x
  store double %val2, double addrspace(3)* %arrayidx2, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
