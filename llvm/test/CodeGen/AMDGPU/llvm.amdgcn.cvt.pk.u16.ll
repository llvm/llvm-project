; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}s_cvt_pk_u16_u32:
; GCN-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s[4:5], 0x{{9|24}}
; GCN: v_mov_b32_e32 [[VY:v[0-9]+]], s[[#LOAD + 3]]
; SI: v_cvt_pk_u16_u32_e32 v{{[0-9]+}}, s[[#LOAD + 2]], [[VY]]
; VI: v_cvt_pk_u16_u32 v{{[0-9]+}}, s[[#LOAD + 2]], [[VY]]
define amdgpu_kernel void @s_cvt_pk_u16_u32(ptr addrspace(1) %out, i32 %x, i32 %y) #0 {
  %result = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %x, i32 %y)
  %r = bitcast <2 x i16> %result to i32
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_cvt_pk_u16_samereg_i32:
; GCN: s_load_dword [[X:s[0-9]+]]
; GCN: v_cvt_pk_u16_u32{{(_e64)*}} v{{[0-9]+}}, [[X]], [[X]]
define amdgpu_kernel void @s_cvt_pk_u16_samereg_i32(ptr addrspace(1) %out, i32 %x) #0 {
  %result = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %x, i32 %x)
  %r = bitcast <2 x i16> %result to i32
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u16_u32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[B:v[0-9]+]]
; SI: v_cvt_pk_u16_u32_e32 v{{[0-9]+}}, [[A]], [[B]]
; VI: v_cvt_pk_u16_u32 v{{[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @v_cvt_pk_u16_u32(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i32, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile i32, ptr addrspace(1) %a.gep
  %b = load volatile i32, ptr addrspace(1) %b.gep
  %cvt = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %a, i32 %b)
  %r = bitcast <2 x i16> %cvt to i32
  store i32 %r, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u16_u32_reg_imm:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_pk_u16_u32{{(_e64)*}} v{{[0-9]+}}, [[A]], 1
define amdgpu_kernel void @v_cvt_pk_u16_u32_reg_imm(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile i32, ptr addrspace(1) %a.gep
  %cvt = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %a, i32 1)
  %r = bitcast <2 x i16> %cvt to i32
  store i32 %r, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u16_u32_imm_reg:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; SI: v_cvt_pk_u16_u32_e32 v{{[0-9]+}}, 1, [[A]]
; VI: v_cvt_pk_u16_u32 v{{[0-9]+}}, 1, [[A]]
define amdgpu_kernel void @v_cvt_pk_u16_u32_imm_reg(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i32, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load volatile i32, ptr addrspace(1) %a.gep
  %cvt = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 1, i32 %a)
  %r = bitcast <2 x i16> %cvt to i32
  store i32 %r, ptr addrspace(1) %out.gep
  ret void
}

declare <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32, i32) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1


attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
