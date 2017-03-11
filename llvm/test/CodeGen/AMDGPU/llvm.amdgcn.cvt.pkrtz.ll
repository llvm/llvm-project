; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}s_cvt_pkrtz_v2f16_f32:
; GCN-DAG: s_load_dword [[X:s[0-9]+]], s[0:1], 0x{{b|2c}}
; GCN-DAG: s_load_dword [[SY:s[0-9]+]], s[0:1], 0x{{c|30}}
; GCN: v_mov_b32_e32 [[VY:v[0-9]+]], [[SY]]
; SI: v_cvt_pkrtz_f16_f32_e32 v{{[0-9]+}}, [[X]], [[VY]]
; VI: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, [[X]], [[VY]]
define void @s_cvt_pkrtz_v2f16_f32(<2 x half> addrspace(1)* %out, float %x, float %y) #0 {
  %result = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %x, float %y)
  store <2 x half> %result, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_cvt_pkrtz_samereg_v2f16_f32:
; GCN: s_load_dword [[X:s[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, [[X]], [[X]]
define void @s_cvt_pkrtz_samereg_v2f16_f32(<2 x half> addrspace(1)* %out, float %x) #0 {
  %result = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %x, float %x)
  store <2 x half> %result, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_cvt_pkrtz_undef_undef:
; GCN-NEXT: ; BB#0
; GCN-NEXT: s_endpgm
define void @s_cvt_pkrtz_undef_undef(<2 x half> addrspace(1)* %out) #0 {
  %result = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float undef, float undef)
  store <2 x half> %result, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; SI: v_cvt_pkrtz_f16_f32_e32 v{{[0-9]+}}, [[A]], [[B]]
; VI: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, [[A]], [[B]]
define void @v_cvt_pkrtz_v2f16_f32(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %a, float %b)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_reg_imm:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, [[A]], 1.0
define void @v_cvt_pkrtz_v2f16_f32_reg_imm(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %a, float 1.0)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_imm_reg:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; SI: v_cvt_pkrtz_f16_f32_e32 v{{[0-9]+}}, 1.0, [[A]]
; VI: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, 1.0, [[A]]
define void @v_cvt_pkrtz_v2f16_f32_imm_reg(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float 1.0, float %a)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_fneg_lo:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, -[[A]], [[B]]
define void @v_cvt_pkrtz_v2f16_f32_fneg_lo(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %neg.a = fsub float -0.0, %a
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %neg.a, float %b)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_fneg_hi:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, [[A]], -[[B]]
define void @v_cvt_pkrtz_v2f16_f32_fneg_hi(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %neg.b = fsub float -0.0, %b
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %a, float %neg.b)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_fneg_lo_hi:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, -[[A]], -[[B]]
define void @v_cvt_pkrtz_v2f16_f32_fneg_lo_hi(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %neg.a = fsub float -0.0, %a
  %neg.b = fsub float -0.0, %b
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %neg.a, float %neg.b)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pkrtz_v2f16_f32_fneg_fabs_lo_fneg_hi:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_cvt_pkrtz_f16_f32_e64 v{{[0-9]+}}, -|[[A]]|, -[[B]]
define void @v_cvt_pkrtz_v2f16_f32_fneg_fabs_lo_fneg_hi(<2 x half> addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fabs.a = call float @llvm.fabs.f32(float %a)
  %neg.fabs.a = fsub float -0.0, %fabs.a
  %neg.b = fsub float -0.0, %b
  %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %neg.fabs.a, float %neg.b)
  store <2 x half> %cvt, <2 x half> addrspace(1)* %out.gep
  ret void
}

declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1
declare float @llvm.fabs.f32(float) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1


attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
