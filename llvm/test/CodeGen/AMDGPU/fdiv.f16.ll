; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS %s
; RUN: llc -march=amdgcn -mcpu=fiji -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS,GFX11 %s

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

; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[LHS:v[0-9]+]]
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[RHS:v[0-9]+]]

; GFX8PLUS-DAG: v_cvt_f32_f16_e32 [[CVT_LHS:v[0-9]+]], [[LHS]]
; GFX8PLUS-DAG: v_cvt_f32_f16_e32 [[CVT_RHS:v[0-9]+]], [[RHS]]

; GFX8PLUS-DAG: v_rcp_f32_e32 [[RCP_RHS:v[0-9]+]], [[CVT_RHS]]
; GFX8PLUS: v_mul_f32_e32 [[MUL:v[0-9]+]], [[CVT_LHS]], [[RCP_RHS]]
; GFX8PLUS: v_cvt_f16_f32_e32 [[CVT_BACK:v[0-9]+]], [[MUL]]
; GFX8PLUS: v_div_fixup_f16 [[RESULT:v[0-9]+]], [[CVT_BACK]], [[RHS]], [[LHS]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
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
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_rcp_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8PLUS-NOT: [[RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half 1.0, %b.val, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16_abs:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_rcp_f16_e64 [[RESULT:v[0-9]+]], |[[VAL]]|
; GFX8PLUS-NOT: [RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_abs(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.abs = call half @llvm.fabs.f16(half %b.val)
  %r.val = fdiv half 1.0, %b.abs, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; We could not do 1/b -> rcp_f16(b) under !fpmath < 1ulp.

; GCN-LABEL: {{^}}reciprocal_f16_rounded:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL16:v[0-9]+]], v{{.+}}
; GFX8PLUS: v_cvt_f32_f16_e32 [[CVT_TO32:v[0-9]+]], [[VAL16]]
; GFX8PLUS: v_rcp_f32_e32 [[RCP32:v[0-9]+]], [[CVT_TO32]]
; GFX8PLUS: v_cvt_f16_f32_e32 [[CVT_BACK16:v[0-9]+]], [[RCP32]]
; GFX8PLUS: v_div_fixup_f16 [[RESULT:v[0-9]+]], [[CVT_BACK16]], [[VAL16]], 1.0
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @reciprocal_f16_rounded(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
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

; GCN-LABEL: {{^}}v_rcp_f16_afn:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_rcp_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8PLUS-NOT: [[RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_afn(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv afn half 1.0, %b.val, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rcp_f16_neg:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_rcp_f16_e64 [[RESULT:v[0-9]+]], -[[VAL]]
; GFX8PLUS-NOT: [RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rcp_f16_neg(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv half -1.0, %b.val, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rsq_f16:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_rsq_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GFX8PLUS-NOT: [RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rsq_f16(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.sqrt = call half @llvm.sqrt.f16(half %b.val)
  %r.val = fdiv half 1.0, %b.sqrt, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_rsq_f16_neg:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[VAL:v[0-9]+]]
; GFX8PLUS-NOT: [[VAL]]
; GFX8PLUS: v_sqrt_f16_e32 [[SQRT:v[0-9]+]], [[VAL]]
; GFX11-NEXT: s_waitcnt_depctr 0xfff
; GFX8PLUS-NEXT: v_rcp_f16_e64 [[RESULT:v[0-9]+]], -[[SQRT]]
; GFX8PLUS-NOT: [RESULT]]
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_rsq_f16_neg(half addrspace(1)* %r, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %b.sqrt = call half @llvm.sqrt.f16(half %b.val)
  %r.val = fdiv half -1.0, %b.sqrt, !fpmath !0
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_f16_afn:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[LHS:v[0-9]+]]
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[RHS:v[0-9]+]]

; GFX8PLUS: v_rcp_f16_e32 [[RCP:v[0-9]+]], [[RHS]]
; GFX8PLUS: v_mul_f16_e32 [[RESULT:v[0-9]+]], [[LHS]], [[RCP]]

; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
define amdgpu_kernel void @v_fdiv_f16_afn(half addrspace(1)* %r, half addrspace(1)* %a, half addrspace(1)* %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds half, half addrspace(1)* %a, i64 %tid.ext
  %gep.b = getelementptr inbounds half, half addrspace(1)* %b, i64 %tid.ext
  %gep.r = getelementptr inbounds half, half addrspace(1)* %r, i64 %tid.ext
  %a.val = load volatile half, half addrspace(1)* %gep.a
  %b.val = load volatile half, half addrspace(1)* %gep.b
  %r.val = fdiv afn half %a.val, %b.val
  store half %r.val, half addrspace(1)* %gep.r
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_f16_unsafe:
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[LHS:v[0-9]+]]
; GFX8PLUS: {{flat|global}}_load_{{ushort|u16}} [[RHS:v[0-9]+]]

; GFX8PLUS: v_rcp_f16_e32 [[RCP:v[0-9]+]], [[RHS]]
; GFX8PLUS: v_mul_f16_e32 [[RESULT:v[0-9]+]], [[LHS]], [[RCP]]

; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.+}}, [[RESULT]]
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

; GCN-LABEL: {{^}}div_afn_2_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0.5, v{{[0-9]+}}

; GFX8PLUS: v_mul_f16_e32 [[MUL:v[0-9]+]], 0.5, v{{[0-9]+}}
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.*}}, [[MUL]]
define amdgpu_kernel void @div_afn_2_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv afn half %x, 2.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}div_afn_k_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0x3dcccccd, v{{[0-9]+}}

; GFX8PLUS: v_mul_f16_e32 [[MUL:v[0-9]+]], 0x2e66, v{{[0-9]+}}
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.*}}, [[MUL]]
define amdgpu_kernel void @div_afn_k_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv afn half %x, 10.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}div_afn_neg_k_x_pat_f16:
; SI: v_mul_f32_e32 v{{[0-9]+}}, 0xbdcccccd, v{{[0-9]+}}

; GFX8PLUS: v_mul_f16_e32 [[MUL:v[0-9]+]], 0xae66, v{{[0-9]+}}
; GFX8PLUS: {{flat|global}}_store_{{short|b16}} v{{.*}}, [[MUL]]
define amdgpu_kernel void @div_afn_neg_k_x_pat_f16(half addrspace(1)* %out) #0 {
  %x = load half, half addrspace(1)* undef
  %rcp = fdiv afn half %x, -10.0
  store half %rcp, half addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2
declare half @llvm.sqrt.f16(half) #2
declare half @llvm.fabs.f16(half) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "unsafe-fp-math"="true" }

!0 = !{float 2.500000e+00}
