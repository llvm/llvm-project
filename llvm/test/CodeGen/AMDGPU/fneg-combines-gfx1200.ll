; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -start-before=amdgpu-unify-divergent-exit-nodes --verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-SDAG %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1200 -start-before=amdgpu-unify-divergent-exit-nodes --verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-GISEL %s

; --------------------------------------------------------------------------------
; fminimum tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_maximum_f32 [[RESULT:v[0-9]+]], -[[A]], -[[B]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %min = call float @llvm.minimum.f32(float %a, float %b)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_maximum_f32 [[RESULT:v[0-9]+]], -[[A]], -[[A]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_self_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.minimum.f32(float %a, float %a)
  %min.fneg = fneg float %min
  store float %min.fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_maximum_f32 [[RESULT:v[0-9]+]], -[[A]], -4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_posk_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.minimum.f32(float %a, float 4.0)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_maximum_f32 [[RESULT:v[0-9]+]], -[[A]], 4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_negk_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.minimum.f32(float %a, float -4.0)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_minimum_f32 [[RESULT:v[0-9]+]], [[A]], 0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_0_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.minimum.f32(float %a, float 0.0)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_minimum_foldable_use_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_minimum_f32 [[MIN:v[0-9]+]], [[A]], 0
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MIN]], [[B]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_0_minimum_foldable_use_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %min = call float @llvm.minimum.f32(float %a, float 0.0)
  %fneg = fneg float %min
  %mul = fmul float %fneg, %b
  store float %mul, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_minimum_multi_use_minimum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_maximum_f32 [[MAX0:v[0-9]+]], -[[A]], -[[B]]
; GCN-SDAG:  v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MAX0]]
; GCN-GISEL: v_mul_f32_e64 [[MUL1:v[0-9]+]], -[[MAX0]], 4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[MAX0]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[MUL1]]
define void @v_fneg_minimum_multi_use_minimum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %min = call float @llvm.minimum.f32(float %a, float %b)
  %fneg = fneg float %min
  %use1 = fmul float %min, 4.0
  store volatile float %fneg, ptr addrspace(1) %out
  store volatile float %use1, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------------
; fmaximum tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_minimum_f32 [[RESULT:v[0-9]+]], -[[A]], -[[B]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %min = call float @llvm.maximum.f32(float %a, float %b)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_minimum_f32 [[RESULT:v[0-9]+]], -[[A]], -[[A]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_self_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.maximum.f32(float %a, float %a)
  %min.fneg = fneg float %min
  store float %min.fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_minimum_f32 [[RESULT:v[0-9]+]], -[[A]], -4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_posk_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.maximum.f32(float %a, float 4.0)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_minimum_f32 [[RESULT:v[0-9]+]], -[[A]], 4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_negk_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %min = call float @llvm.maximum.f32(float %a, float -4.0)
  %fneg = fneg float %min
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: v_maximum_f32 [[RESULT:v[0-9]+]], [[A]], 0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_0_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %max = call float @llvm.maximum.f32(float %a, float 0.0)
  %fneg = fneg float %max
  store float %fneg, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_maximum_foldable_use_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_maximum_f32 [[MAX:v[0-9]+]], [[A]], 0
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MAX]], [[B]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[RESULT]]
define void @v_fneg_0_maximum_foldable_use_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %max = call float @llvm.maximum.f32(float %a, float 0.0)
  %fneg = fneg float %max
  %mul = fmul float %fneg, %b
  store float %mul, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_maximum_multi_use_maximum_f32:
; GCN: global_load_b32 [[A:v[0-9]+]]
; GCN: global_load_b32 [[B:v[0-9]+]]
; GCN: v_minimum_f32 [[MAX0:v[0-9]+]], -[[A]], -[[B]]
; GCN-SDAG:  v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MAX0]]
; GCN-GISEL: v_mul_f32_e64 [[MUL1:v[0-9]+]], -[[MAX0]], 4.0
; GCN: global_store_b32 v[{{[0-9:]+}}], [[MAX0]]
; GCN: global_store_b32 v[{{[0-9:]+}}], [[MUL1]]
define void @v_fneg_maximum_multi_use_maximum_f32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile float, ptr addrspace(1) %a.gep
  %b = load volatile float, ptr addrspace(1) %b.gep
  %min = call float @llvm.maximum.f32(float %a, float %b)
  %fneg = fneg float %min
  %use1 = fmul float %min, 4.0
  store volatile float %fneg, ptr addrspace(1) %out
  store volatile float %use1, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
