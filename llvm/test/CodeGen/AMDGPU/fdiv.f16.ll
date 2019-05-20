; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -mattr=+fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX8_9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -mattr=-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX8_9 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX8_9 %s

; Make sure fdiv is promoted to f32.

; GCN-LABEL: {{^}}v_fdiv_f16
; SI:     v_cvt_f32_f16
; SI:     v_cvt_f32_f16
; SI:     v_div_scale_f32
; SI-DAG: v_div_scale_f32
; SI-DAG: v_rcp_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_mul_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_div_fmas_f32
; SI:     v_div_fixup_f32
; SI:     v_cvt_f16_f32

; GFX8_9: {{flat|global}}_load_ushort [[LHS:v[0-9]+]]
; GFX8_9: {{flat|global}}_load_ushort [[RHS:v[0-9]+]]

; GFX8_9-DAG: v_cvt_f32_f16_e32 [[CVT_LHS:v[0-9]+]], [[LHS]]
; GFX8_9-DAG: v_cvt_f32_f16_e32 [[CVT_RHS:v[0-9]+]], [[RHS]]

; GFX8_9-DAG: v_rcp_f32_e32 [[RCP_RHS:v[0-9]+]], [[CVT_RHS]]
; GFX8_9: v_mul_f32_e32 [[MUL:v[0-9]+]], [[CVT_LHS]], [[RCP_RHS]]
; GFX8_9: v_cvt_f16_f32_e32 [[CVT_BACK:v[0-9]+]], [[MUL]]
; GFX8_9: v_div_fixup_f16 [[RESULT:v[0-9]+]], [[CVT_BACK]], [[RHS]], [[LHS]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fdiv_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds half, half addrspace(1)* %a, i64 %tid.ext
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %a.val = load volatile half, half addrspace(1)* %gep.a
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half %a.val, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_rcp_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8_9-NOT: [[RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half 1.0, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16_abs:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_rcp_f16_e64 [[RESULT:v[0-9]+]], |[[VAL]]|
; GFX8_9-NOT: [RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_abs(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.abs = call half @llvm.fabs.f16(half %b.val)
  %r.val = fdiv half 1.0, %b.abs
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16_arcp:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_rcp_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8_9-NOT: [[RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_arcp(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv arcp half 1.0, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16_neg:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_rcp_f16_e64 [[RESULT:v[0-9]+]], -[[VAL]]
; GFX8_9-NOT: [RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_neg(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half -1.0, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rsq_f16:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_rsq_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8_9-NOT: [RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rsq_f16(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.sqrt = call half @llvm.sqrt.f16(half %b.val)
  %r.val = fdiv half 1.0, %b.sqrt
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rsq_f16_neg:
; GFX8_9: {{flat|global}}_load_ushort [[VAL:v[0-9]+]]
; GFX8_9-NOT: [[VAL]]
; GFX8_9: v_sqrt_f16_e32 [[SQRT:v[0-9]+]], [[VAL]]
; GFX8_9-NEXT: v_rcp_f16_e64 [[RESULT:v[0-9]+]], -[[SQRT]]
; GFX8_9-NOT: [RESULT]]
; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_rsq_f16_neg(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.sqrt = call half @llvm.sqrt.f16(half %b.val)
  %r.val = fdiv half -1.0, %b.sqrt
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_f16_arcp:
; GFX8_9: {{flat|global}}_load_ushort [[LHS:v[0-9]+]]
; GFX8_9: {{flat|global}}_load_ushort [[RHS:v[0-9]+]]

; GFX8_9: v_rcp_f16_e32 [[RCP:v[0-9]+]], [[RHS]]
; GFX8_9: v_mul_f16_e32 [[RESULT:v[0-9]+]], [[LHS]], [[RCP]]

; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fdiv_f16_arcp(half addrspace(1)* %r, half addrspace(1)* %a, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds half, half addrspace(1)* %a, i64 %tid.ext
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %a.val = load volatile half, half addrspace(1)* %gep.a
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv arcp half %a.val, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_f16_unsafe:
; GFX8_9: {{flat|global}}_load_ushort [[LHS:v[0-9]+]]
; GFX8_9: {{flat|global}}_load_ushort [[RHS:v[0-9]+]]

; GFX8_9: v_rcp_f16_e32 [[RCP:v[0-9]+]], [[RHS]]
; GFX8_9: v_mul_f16_e32 [[RESULT:v[0-9]+]], [[LHS]], [[RCP]]

; GFX8_9: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fdiv_f16_unsafe(half addrspace(1)* %r, half addrspace(1)* %a, half addrspace(1)* %b) #2 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds half, half addrspace(1)* %a, i64 %tid.ext
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %a.val = load volatile half, half addrspace(1)* %gep.a
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half %a.val, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_2_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0.5, v{{[0-9]+}}

; GFX8_9: v_mul_f16_e32 [[MUL:v[0-9]+]], 0.5, v{{[0-9]+}}
; GFX8_9: buffer_store_short [[MUL]]
define amdgpu_kernel void @div_arcp_2_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv arcp half %x, 2.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_k_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0x3dccc000, v{{[0-9]+}}

; GFX8_9: v_mul_f16_e32 [[MUL:v[0-9]+]], 0x2e66, v{{[0-9]+}}
; GFX8_9: buffer_store_short [[MUL]]
define amdgpu_kernel void @div_arcp_k_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv arcp half %x, 10.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_neg_k_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0xbdccc000, v{{[0-9]+}}

; GFX8_9: v_mul_f16_e32 [[MUL:v[0-9]+]], 0xae66, v{{[0-9]+}}
; GFX8_9: buffer_store_short [[MUL]]
define amdgpu_kernel void @div_arcp_neg_k_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv arcp half %x, -10.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare half @llvm.sqrt.f16(half) #1
declare half @llvm.fabs.f16(half) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "unsafe-fp-math"="true" }
