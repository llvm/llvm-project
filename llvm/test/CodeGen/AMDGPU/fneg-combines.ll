; RUN: llc -march=amdgcn -mcpu=hawaii -start-after=sink -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-SAFE,SI %s
; RUN: llc -enable-no-signed-zeros-fp-math -march=amdgcn -mcpu=hawaii -mattr=+flat-for-global -start-after=sink -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-NSZ,SI %s

; RUN: llc -march=amdgcn -mcpu=fiji -start-after=sink --verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-SAFE,VI %s
; RUN: llc -enable-no-signed-zeros-fp-math -march=amdgcn -mcpu=fiji -start-after=sink -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-NSZ,VI %s

; --------------------------------------------------------------------------------
; fadd tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[A]], [[B]]
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fneg float %add
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_ADD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_add_store_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fneg float %add
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %add, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_multi_use_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-SAFE: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]

; GCN-NSZ: v_sub_f32_e64 [[NEG_ADD:v[0-9]+]], -[[A]], [[B]]
; GCN-NSZ-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[NEG_ADD]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_ADD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_add_multi_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fneg float %add
  %use1 = fmul float %add, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e32
; GCN-SAFE: v_xor_b32_e32 [[ADD:v[0-9]+]], 0x80000000,

; GCN-NSZ: v_sub_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_add_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %add = fadd float %fneg.a, %b
  %fneg = fneg float %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_add_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fneg float %b
  %add = fadd float %a, %fneg.b
  %fneg = fneg float %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e64 [[ADD:v[0-9]+]], -[[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_add_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %fneg.b = fneg float %b
  %add = fadd float %fneg.a, %fneg.b
  %fneg = fneg float %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_fneg_x_f32:
; GCN-SAFE-DAG: s_brev_b32 [[SIGNBIT:s[0-9]+]], 1{{$}}
; GCN-DAG: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_xor_b32_e32 [[NEG_A:v[0-9]+]], [[SIGNBIT]], [[A]]
; GCN-SAFE: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], [[SIGNBIT]], [[ADD]]

; GCN-NSZ-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-NSZ-DAG: v_sub_f32_e32 [[NEG_ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_ADD]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_add_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %add = fadd float %fneg.a, %b
  %fneg = fneg float %add
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-SAFE-DAG: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-SAFE-DAG: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ-DAG: v_sub_f32_e32 [[NEG_ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_ADD]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_add_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %add = fadd float %fneg.a, %b
  %fneg = fneg float %add
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; This one asserted with -enable-no-signed-zeros-fp-math
; GCN-LABEL: {{^}}fneg_fadd_0:
; GCN-SAFE-DAG: v_mad_f32 [[A:v[0-9]+]],
; GCN-SAFE-DAG: v_cmp_ngt_f32_e32 {{.*}}, [[A]]
; GCN-SAFE-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, -[[A]]
define amdgpu_ps float @fneg_fadd_0(float inreg %tmp2, float inreg %tmp6, <4 x i32> %arg) local_unnamed_addr #0 {
.entry:
  %tmp7 = fdiv float 1.000000e+00, %tmp6
  %tmp8 = fmul float 0.000000e+00, %tmp7
  %tmp9 = fmul reassoc nnan arcp contract float 0.000000e+00, %tmp8
  %.i188 = fadd float %tmp9, 0.000000e+00
  %tmp10 = fcmp uge float %.i188, %tmp2
  %tmp11 = fneg float %.i188
  %.i092 = select i1 %tmp10, float %tmp2, float %tmp11
  %tmp12 = fcmp ule float %.i092, 0.000000e+00
  %.i198 = select i1 %tmp12, float 0.000000e+00, float 0x7FF8000000000000
  ret float %.i198
}

; This is a workaround because -enable-no-signed-zeros-fp-math does not set up
; function attribute unsafe-fp-math automatically. Combine with the previous test
; when that is done.
; GCN-LABEL: {{^}}fneg_fadd_0_nsz:
; GCN-NSZ-DAG: v_rcp_f32_e32 [[A:v[0-9]+]],
; GCN-NSZ-DAG: v_mov_b32_e32 [[B:v[0-9]+]],
; GCN-NSZ-DAG: v_mov_b32_e32 [[C:v[0-9]+]],
; GCN-NSZ-DAG: v_mul_f32_e32 [[D:v[0-9]+]],
; GCN-NSZ-DAG: v_cmp_nlt_f32_e64 {{.*}}, -[[D]]
define amdgpu_ps float @fneg_fadd_0_nsz(float inreg %tmp2, float inreg %tmp6, <4 x i32> %arg) local_unnamed_addr #2 {
.entry:
  %tmp7 = fdiv afn float 1.000000e+00, %tmp6
  %tmp8 = fmul float 0.000000e+00, %tmp7
  %tmp9 = fmul reassoc nnan arcp contract float 0.000000e+00, %tmp8
  %.i188 = fadd float %tmp9, 0.000000e+00
  %tmp10 = fcmp uge float %.i188, %tmp2
  %tmp11 = fneg float %.i188
  %.i092 = select i1 %tmp10, float %tmp2, float %tmp11
  %tmp12 = fcmp ule float %.i092, 0.000000e+00
  %.i198 = select i1 %tmp12, float 0.000000e+00, float 0x7FF8000000000000
  ret float %.i198
}

; --------------------------------------------------------------------------------
; fmul tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_store_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[MUL0:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MUL0]]

; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_mul_multi_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  %use1 = fmul float %mul, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = fmul float %fneg.a, %b
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fneg float %b
  %mul = fmul float %a, %fneg.b
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %fneg.b = fneg float %b
  %mul = fmul float %fneg.a, %fneg.b
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[A]], [[B]]

; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
define amdgpu_kernel void @v_fneg_mul_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = fmul float %fneg.a, %b
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_mul_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = fmul float %fneg.a, %b
  %fneg = fneg float %mul
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fminnum tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_B:v[0-9]+]], -1.0, [[B]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_max_f32_e64 v0, -v0, -v1
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_minnum_f32_no_ieee(float %a, float %b) #0 {
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fneg float %min
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_self_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_self_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float %a, float %a)
  %min.fneg = fneg float %min
  store float %min.fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_max_f32_e64 v0, -v0, -v0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_self_minnum_f32_no_ieee(float %a) #0 {
  %min = call float @llvm.minnum.f32(float %a, float %a)
  %min.fneg = fneg float %min
  ret float %min.fneg
}

; GCN-LABEL: {{^}}v_fneg_posk_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], -4.0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_posk_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 4.0, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_max_f32_e64 v0, -v0, -4.0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_posk_minnum_f32_no_ieee(float %a) #0 {
  %min = call float @llvm.minnum.f32(float 4.0, float %a)
  %fneg = fneg float %min
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_negk_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], 4.0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_negk_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float -4.0, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_max_f32_e64 v0, -v0, 4.0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_negk_minnum_f32_no_ieee(float %a) #0 {
  %min = call float @llvm.minnum.f32(float -4.0, float %a)
  %fneg = fneg float %min
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_0_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], 0, [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_0_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 0.0, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_neg0_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float -0.0, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_inv2pi_minnum_f32:
; GCN-DAG: {{buffer|flat}}_load_dword [[A:v[0-9]+]]

; SI-DAG: v_mul_f32_e32 [[QUIET_NEG:v[0-9]+]], -1.0, [[A]]
; SI: v_max_f32_e32 [[RESULT:v[0-9]+]], 0xbe22f983, [[QUIET_NEG]]

; VI: v_mul_f32_e32 [[QUIET:v[0-9]+]], 1.0, [[A]]
; VI: v_min_f32_e32 [[MAX:v[0-9]+]], 0.15915494, [[QUIET]]
; VI: v_xor_b32_e32 [[RESULT:v[0-9]+]], 0x80000000, [[MAX]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_inv2pi_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 0x3FC45F3060000000, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg_inv2pi_minnum_f32:
; GCN-DAG: {{buffer|flat}}_load_dword [[A:v[0-9]+]]

; SI: v_mul_f32_e32 [[NEG_QUIET:v[0-9]+]], -1.0, [[A]]
; SI: v_max_f32_e32 [[RESULT:v[0-9]+]], 0x3e22f983, [[NEG_QUIET]]

; VI: v_mul_f32_e32 [[NEG_QUIET:v[0-9]+]], -1.0, [[A]]
; VI: v_max_f32_e32 [[RESULT:v[0-9]+]], 0.15915494, [[NEG_QUIET]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_neg_inv2pi_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 0xBFC45F3060000000, float %a)
  %fneg = fneg float %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_inv2pi_minnum_f16:
; GCN-DAG: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]

; SI: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -[[A]]
; SI: v_max_f32_e32 [[MAX:v[0-9]+]], 0xbe230000, [[CVT]]
; SI: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[MAX]]

; VI: v_max_f16_e32 [[QUIET:v[0-9]+]], [[A]], [[A]]
; VI: v_min_f16_e32 [[MAX:v[0-9]+]], 0.15915494, [[QUIET]]
; VI: v_xor_b32_e32 [[RESULT:v[0-9]+]], 0x8000, [[MAX]]

; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_inv2pi_minnum_f16(half addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds half, half addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %a.gep
  %min = call half @llvm.minnum.f16(half 0xH3118, half %a)
  %fneg = fsub half -0.000000e+00, %min
  store half %fneg, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg_inv2pi_minnum_f16:
; GCN-DAG: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]

; SI: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -[[A]]
; SI: v_max_f32_e32 [[MAX:v[0-9]+]], 0x3e230000, [[CVT]]
; SI: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[MAX]]

; VI: v_max_f16_e64 [[NEG_QUIET:v[0-9]+]], -[[A]], -[[A]]
; VI: v_max_f16_e32 [[RESULT:v[0-9]+]], 0.15915494, [[NEG_QUIET]]

; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_neg_inv2pi_minnum_f16(half addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds half, half addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %a.gep
  %min = call half @llvm.minnum.f16(half 0xHB118, half %a)
  %fneg = fsub half -0.000000e+00, %min
  store half %fneg, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_inv2pi_minnum_f64:
; GCN-DAG: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]

; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0xbfc45f30
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0x6dc9c882
; SI-DAG: v_max_f64 [[NEG_QUIET:v\[[0-9]+:[0-9]+\]]], -[[A]], -[[A]]
; SI: v_max_f64 v{{\[}}[[RESULT_LO:[0-9]+]]:[[RESULT_HI:[0-9]+]]{{\]}}, [[NEG_QUIET]], s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}

; VI: v_min_f64 v{{\[}}[[RESULT_LO:[0-9]+]]:[[RESULT_HI:[0-9]+]]{{\]}}, [[A]], 0.15915494
; VI: v_xor_b32_e32 v[[RESULT_HI]], 0x80000000, v[[RESULT_HI]]

; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define amdgpu_kernel void @v_fneg_inv2pi_minnum_f64(double addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %min = call double @llvm.minnum.f64(double 0x3fc45f306dc9c882, double %a)
  %fneg = fsub double -0.000000e+00, %min
  store double %fneg, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg_inv2pi_minnum_f64:
; GCN-DAG: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]

; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0x3fc45f30
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0x6dc9c882
; SI-DAG: v_max_f64 [[NEG_QUIET:v\[[0-9]+:[0-9]+\]]], -[[A]], -[[A]]
; SI: v_max_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[NEG_QUIET]], s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}

; VI: v_max_f64 [[NEG_QUIET:v\[[0-9]+:[0-9]+\]]], -[[A]], -[[A]]
; VI: v_max_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[NEG_QUIET]], 0.15915494

; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_neg_inv2pi_minnum_f64(double addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %min = call double @llvm.minnum.f64(double 0xbfc45f306dc9c882, double %a)
  %fneg = fsub double -0.000000e+00, %min
  store double %fneg, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_max_f32_e64 v0, -v0, 0{{$}}
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_neg0_minnum_f32_no_ieee(float %a) #0 {
  %min = call float @llvm.minnum.f32(float -0.0, float %a)
  %fneg = fneg float %min
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_0_minnum_foldable_use_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_A:v[0-9]+]], 1.0, [[A]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], 0, [[QUIET_A]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MIN]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_0_minnum_foldable_use_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float 0.0, float %a)
  %fneg = fneg float %min
  %mul = fmul float %fneg, %b
  store float %mul, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_inv2pi_minnum_foldable_use_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI: v_mul_f32_e32 [[QUIET_NEG:v[0-9]+]], -1.0, [[A]]

; SI: v_max_f32_e32 [[MIN:v[0-9]+]], 0xbe22f983, [[QUIET_NEG]]
; SI: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[MIN]], [[B]]

; VI: v_mul_f32_e32 [[QUIET:v[0-9]+]], 1.0, [[A]]
; VI: v_min_f32_e32 [[MIN:v[0-9]+]], 0.15915494, [[QUIET]]
; VI: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MIN]], [[B]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_inv2pi_minnum_foldable_use_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float 0x3FC45F3060000000, float %a)
  %fneg = fneg float %min
  %mul = fmul float %fneg, %b
  store float %mul, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_minnum_foldable_use_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], 0, v0
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MIN]], v1
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_0_minnum_foldable_use_f32_no_ieee(float %a, float %b) #0 {
  %min = call float @llvm.minnum.f32(float 0.0, float %a)
  %fneg = fneg float %min
  %mul = fmul float %fneg, %b
  ret float %mul
}

; GCN-LABEL: {{^}}v_fneg_minnum_multi_use_minnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_B:v[0-9]+]], -1.0, [[B]]
; GCN: v_max_f32_e32 [[MAX0:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MAX0]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MAX0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_minnum_multi_use_minnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fneg float %min
  %use1 = fmul float %min, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_minnum_multi_use_minnum_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_max_f32_e64 v0, -v0, -v1
; GCN-NEXT: v_mul_f32_e32 v1, -4.0, v0
; GCN-NEXT: ; return
define amdgpu_ps <2 x float> @v_fneg_minnum_multi_use_minnum_f32_no_ieee(float %a, float %b) #0 {
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fneg float %min
  %use1 = fmul float %min, 4.0
  %ins0 = insertelement <2 x float> undef, float %fneg, i32 0
  %ins1 = insertelement <2 x float> %ins0, float %use1, i32 1
  ret <2 x float> %ins1
}

; --------------------------------------------------------------------------------
; fmaxnum tests
; --------------------------------------------------------------------------------


; GCN-LABEL: {{^}}v_fneg_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_B:v[0-9]+]], -1.0, [[B]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %max = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fneg float %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_min_f32_e64 v0, -v0, -v1
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_maxnum_f32_no_ieee(float %a, float %b) #0 {
  %max = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fneg float %max
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_self_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_self_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float %a, float %a)
  %max.fneg = fneg float %max
  store float %max.fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_min_f32_e64 v0, -v0, -v0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_self_maxnum_f32_no_ieee(float %a) #0 {
  %max = call float @llvm.maxnum.f32(float %a, float %a)
  %max.fneg = fneg float %max
  ret float %max.fneg
}

; GCN-LABEL: {{^}}v_fneg_posk_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], -4.0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_posk_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float 4.0, float %a)
  %fneg = fneg float %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_min_f32_e64 v0, -v0, -4.0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_posk_maxnum_f32_no_ieee(float %a) #0 {
  %max = call float @llvm.maxnum.f32(float 4.0, float %a)
  %fneg = fneg float %max
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_negk_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], 4.0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_negk_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float -4.0, float %a)
  %fneg = fneg float %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_min_f32_e64 v0, -v0, 4.0
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_negk_maxnum_f32_no_ieee(float %a) #0 {
  %max = call float @llvm.maxnum.f32(float -4.0, float %a)
  %fneg = fneg float %max
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_0_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_0_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float 0.0, float %a)
  %fneg = fneg float %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_NEG_A:v[0-9]+]], -1.0, [[A]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], 0, [[QUIET_NEG_A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_neg0_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float -0.0, float %a)
  %fneg = fneg float %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN: v_min_f32_e64 v0, -v0, 0{{$}}
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_neg0_maxnum_f32_no_ieee(float %a) #0 {
  %max = call float @llvm.maxnum.f32(float -0.0, float %a)
  %fneg = fneg float %max
  ret float %fneg
}

; GCN-LABEL: {{^}}v_fneg_0_maxnum_foldable_use_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[QUIET_A:v[0-9]+]], 1.0, [[A]]
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], 0, [[QUIET_A]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MAX]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_0_maxnum_foldable_use_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %max = call float @llvm.maxnum.f32(float 0.0, float %a)
  %fneg = fneg float %max
  %mul = fmul float %fneg, %b
  store float %mul, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_maxnum_foldable_use_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], 0, v0
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MAX]], v1
; GCN-NEXT: ; return
define amdgpu_ps float @v_fneg_0_maxnum_foldable_use_f32_no_ieee(float %a, float %b) #0 {
  %max = call float @llvm.maxnum.f32(float 0.0, float %a)
  %fneg = fneg float %max
  %mul = fmul float %fneg, %b
  ret float %mul
}

; GCN-LABEL: {{^}}v_fneg_maxnum_multi_use_maxnum_f32_ieee:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_A:v[0-9]+]], -1.0, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_QUIET_B:v[0-9]+]], -1.0, [[B]]
; GCN: v_min_f32_e32 [[MAX0:v[0-9]+]], [[NEG_QUIET_A]], [[NEG_QUIET_B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MAX0]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MAX0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_maxnum_multi_use_maxnum_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %max = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fneg float %max
  %use1 = fmul float %max, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_maxnum_multi_use_maxnum_f32_no_ieee:
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_min_f32_e64 v0, -v0, -v1
; GCN-NEXT: v_mul_f32_e32 v1, -4.0, v0
; GCN-NEXT: ; return
define amdgpu_ps <2 x float> @v_fneg_maxnum_multi_use_maxnum_f32_no_ieee(float %a, float %b) #0 {
  %max = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fneg float %max
  %use1 = fmul float %max, 4.0
  %ins0 = insertelement <2 x float> undef, float %fneg, i32 0
  %ins1 = insertelement <2 x float> %ins0, float %use1, i32 1
  ret <2 x float> %ins1
}

; --------------------------------------------------------------------------------
; fma tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_fma_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[RESULT]]

; GCN-NSZ: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)
  %fneg = fneg float %fma
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_store_use_fma_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN-DAG: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-DAG: v_xor_b32_e32 [[NEG_FMA:v[0-9]+]], 0x80000000, [[FMA]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_FMA]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_fma_store_use_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_multi_use_fma_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_FMA:v[0-9]+]], 0x80000000, [[FMA]]
; GCN-SAFE: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[FMA]]

; GCN-NSZ: v_fma_f32 [[NEG_FMA:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[NEG_FMA]]

; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_FMA]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_fma_multi_use_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)
  %fneg = fneg float %fma
  %use1 = fmul float %fma, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_fneg_x_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], -[[A]], [[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
define amdgpu_kernel void @v_fneg_fma_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.a = fneg float %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_x_fneg_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
define amdgpu_kernel void @v_fneg_fma_x_fneg_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.b = fneg float %b
  %fma = call float @llvm.fma.f32(float %a, float %fneg.b, float %c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_fneg_fneg_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
define amdgpu_kernel void @v_fneg_fma_fneg_fneg_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.a = fneg float %a
  %fneg.b = fneg float %b
  %fma = call float @llvm.fma.f32(float %fneg.a, float %fneg.b, float %c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_fneg_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], -[[A]], [[B]], -[[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
define amdgpu_kernel void @v_fneg_fma_fneg_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.a = fneg float %a
  %fneg.c = fneg float %c
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %fneg.c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_x_y_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-NSZ-SAFE: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NSZ-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], [[C]]
; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
define amdgpu_kernel void @v_fneg_fma_x_y_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.c = fneg float %c
  %fma = call float @llvm.fma.f32(float %a, float %b, float %fneg.c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_store_use_fneg_x_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_xor_b32
; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], -[[A]],
; GCN-SAFE: v_xor_b32

; GCN-NSZ-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-NSZ-DAG: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]

; GCN-NSZ-NOT: [[FMA]]
; GCN-NSZ-NOT: [[NEG_A]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA]]
; GCN-NSZ-NOT: [[NEG_A]]
; GCN-NSZ: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
define amdgpu_kernel void @v_fneg_fma_store_use_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.a = fneg float %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fneg float %fma
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_multi_use_fneg_x_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-SAFE-DAG: v_fma_f32 [[FMA:v[0-9]+]]
; GCN-SAFE-DAG: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ-DAG: v_fma_f32 [[NEG_FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_FMA]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NSZ-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_fma_multi_use_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, float %d) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fneg.a = fneg float %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fneg float %fma
  %use1 = fmul float %fneg.a, %d
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fmad tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_fmad_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_mac_f32_e32 [[C]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[C]]

; GCN-NSZ: v_mad_f32 [[RESULT:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fmad_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fma = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %fneg = fneg float %fma
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fmad_v4f32:

; GCN-NSZ: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; GCN-NSZ: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; GCN-NSZ: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; GCN-NSZ: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
define amdgpu_kernel void @v_fneg_fmad_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %a.ptr, <4 x float> addrspace(1)* %b.ptr, <4 x float> addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %tid.ext
  %a = load volatile <4 x float>, <4 x float> addrspace(1)* %a.gep
  %b = load volatile <4 x float>, <4 x float> addrspace(1)* %b.gep
  %c = load volatile <4 x float>, <4 x float> addrspace(1)* %c.gep
  %fma = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  %fneg = fneg <4 x float> %fma
  store <4 x float> %fneg, <4 x float> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fmad_multi_use_fmad_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_mac_f32_e32 [[C]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_MAD:v[0-9]+]], 0x80000000, [[C]]
; GCN-SAFE-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[C]]

; GCN-NSZ: v_mad_f32 [[NEG_MAD:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[NEG_MAD]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MAD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_fmad_multi_use_fmad_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %fma = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %fneg = fneg float %fma
  %use1 = fmul float %fma, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fp_extend tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_fp_extend_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_f64_f32_e64 [[RESULT:v\[[0-9]+:[0-9]+\]]], -[[A]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_extend_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fpext = fpext float %a to double
  %fneg = fsub double -0.000000e+00, %fpext
  store double %fneg, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_extend_fneg_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_f64_f32_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]]
; GCN: {{buffer|flat}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %fpext = fpext float %fneg.a to double
  %fneg = fsub double -0.000000e+00, %fpext
  store double %fneg, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_extend_store_use_fneg_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f64_f32_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[FNEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FNEG_A]]
define amdgpu_kernel void @v_fneg_fp_extend_store_use_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %fpext = fpext float %fneg.a to double
  %fneg = fsub double -0.000000e+00, %fpext
  store volatile double %fneg, double addrspace(1)* %out.gep
  store volatile float %fneg.a, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_multi_use_fp_extend_fneg_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f64_f32_e32 v{{\[}}[[CVT_LO:[0-9]+]]:[[CVT_HI:[0-9]+]]{{\]}}, [[A]]
; GCN-DAG: v_xor_b32_e32 v[[FNEG_A:[0-9]+]], 0x80000000, v[[CVT_HI]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+}}:[[FNEG_A]]{{\]}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[CVT_LO]]:[[CVT_HI]]{{\]}}
define amdgpu_kernel void @v_fneg_multi_use_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fpext = fpext float %a to double
  %fneg = fsub double -0.000000e+00, %fpext
  store volatile double %fneg, double addrspace(1)* %out.gep
  store volatile double %fpext, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_multi_foldable_use_fp_extend_fneg_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f64_f32_e32 v{{\[}}[[CVT_LO:[0-9]+]]:[[CVT_HI:[0-9]+]]{{\]}}, [[A]]
; GCN-DAG: v_xor_b32_e32 v[[FNEG_A:[0-9]+]], 0x80000000, v[[CVT_HI]]
; GCN-DAG: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[CVT_LO]]:[[CVT_HI]]{{\]}}, 4.0
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+}}:[[FNEG_A]]{{\]}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_multi_foldable_use_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fpext = fpext float %a to double
  %fneg = fsub double -0.000000e+00, %fpext
  %mul = fmul double %fpext, 4.0
  store volatile double %fneg, double addrspace(1)* %out.gep
  store volatile double %mul, double addrspace(1)* %out.gep
  ret void
}

; FIXME: Source modifiers not folded for f16->f32
; GCN-LABEL: {{^}}v_fneg_multi_use_fp_extend_fneg_f16_to_f32:
define amdgpu_kernel void @v_fneg_multi_use_fp_extend_fneg_f16_to_f32(float addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds half, half addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %a.gep
  %fpext = fpext half %a to float
  %fneg = fneg float %fpext
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %fpext, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_multi_foldable_use_fp_extend_fneg_f16_to_f32:
define amdgpu_kernel void @v_fneg_multi_foldable_use_fp_extend_fneg_f16_to_f32(float addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds half, half addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile half, half addrspace(1)* %a.gep
  %fpext = fpext half %a to float
  %fneg = fneg float %fpext
  %mul = fmul float %fpext, 4.0
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %mul, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; fp_round tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_fp_round_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_cvt_f32_f64_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fpround = fptrunc double %a to float
  %fneg = fneg float %fpround
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fneg float %fpround
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_store_use_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 v{{\[}}[[A_LO:[0-9]+]]:[[A_HI:[0-9]+]]{{\]}}
; GCN-DAG: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], v{{\[}}[[A_LO]]:[[A_HI]]{{\]}}
; GCN-DAG: v_xor_b32_e32 v[[NEG_A_HI:[0-9]+]], 0x80000000, v[[A_HI]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[A_LO]]:[[NEG_A_HI]]{{\]}}
define amdgpu_kernel void @v_fneg_fp_round_store_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fneg float %fpround
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile double %fneg.a, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_multi_use_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_mul_f64 [[USE1:v\[[0-9]+:[0-9]+\]]], -[[A]], s{{\[}}

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[USE1]]
define amdgpu_kernel void @v_fneg_fp_round_multi_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr, double %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fneg float %fpround
  %use1 = fmul double %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile double %use1, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_f16_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fpround = fptrunc float %a to half
  %fneg = fsub half -0.000000e+00, %fpround
  store half %fneg, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_fneg_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %fpround = fptrunc float %fneg.a to half
  %fneg = fsub half -0.000000e+00, %fpround
  store half %fneg, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_multi_use_fp_round_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_cvt_f32_f64_e32 [[CVT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG:v[0-9]+]], 0x80000000, [[CVT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[CVT]]
define amdgpu_kernel void @v_fneg_multi_use_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fpround = fptrunc double %a to float
  %fneg = fneg float %fpround
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %fpround, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_store_use_fneg_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
define amdgpu_kernel void @v_fneg_fp_round_store_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %fpround = fptrunc float %fneg.a to half
  %fneg = fsub half -0.000000e+00, %fpround
  store volatile half %fneg, half addrspace(1)* %out.gep
  store volatile float %fneg.a, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_multi_use_fneg_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[USE1:v[0-9]+]], -[[A]], s
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[USE1]]
define amdgpu_kernel void @v_fneg_fp_round_multi_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %fpround = fptrunc float %fneg.a to half
  %fneg = fsub half -0.000000e+00, %fpround
  %use1 = fmul float %fneg.a, %c
  store volatile half %fneg, half addrspace(1)* %out.gep
  store volatile float %use1, float addrspace(1)* undef
  ret void
}

; --------------------------------------------------------------------------------
; rcp tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_rcp_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rcp_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_rcp_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %rcp = call float @llvm.amdgcn.rcp.f32(float %a)
  %fneg = fneg float %rcp
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_rcp_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fneg float %rcp
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_store_use_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
define amdgpu_kernel void @v_fneg_rcp_store_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fneg float %rcp
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %fneg.a, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_multi_use_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_rcp_multi_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fneg float %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fneg float %rcp
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %use1, float addrspace(1)* undef
  ret void
}

; --------------------------------------------------------------------------------
; fmul_legacy tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[RESULT:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fneg float %mul
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_store_use_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL_LEGACY]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_mul_legacy_store_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_multi_use_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: v_mul_legacy_f32_e64 [[MUL:v[0-9]+]], -[[ADD]], 4.0
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @v_fneg_mul_legacy_multi_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fneg float %mul
  %use1 = call float @llvm.amdgcn.fmul.legacy(float %mul, float 4.0)
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fneg float %b
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %fneg.b)
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %fneg.b = fneg float %b
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %fneg.b)
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL_LEGACY]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_A]]
define amdgpu_kernel void @v_fneg_mul_legacy_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fneg float %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_mul_legacy_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[NEG_MUL_LEGACY]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_mul_legacy_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fneg float %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fneg float %mul
  %use1 = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %c)
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; sin tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_sin_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 0xbe22f983, [[A]]
; GCN: v_fract_f32_e32 [[FRACT:v[0-9]+]], [[MUL]]
; GCN: v_sin_f32_e32 [[RESULT:v[0-9]+]], [[FRACT]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %sin = call float @llvm.sin.f32(float %a)
  %fneg = fneg float %sin
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_amdgcn_sin_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_sin_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_amdgcn_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %sin = call float @llvm.amdgcn.sin.f32(float %a)
  %fneg = fneg float %sin
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; ftrunc tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_trunc_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_trunc_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_trunc_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %trunc = call float @llvm.trunc.f32(float %a)
  %fneg = fneg float %trunc
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; fround tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_round_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_trunc_f32_e32
; GCN: v_sub_f32_e32
; GCN: v_cndmask_b32

; GCN-SAFE: v_add_f32_e32 [[ADD:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-SAFE: v_xor_b32_e32 [[RESULT:v[0-9]+]], 0x80000000, [[ADD]]

; GCN-NSZ: v_sub_f32_e64 [[RESULT:v[0-9]+]], -v{{[0-9]+}}, v{{[0-9]+}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_round_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %round = call float @llvm.round.f32(float %a)
  %fneg = fneg float %round
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; rint tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_rint_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rndne_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_rint_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %rint = call float @llvm.rint.f32(float %a)
  %fneg = fneg float %rint
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; nearbyint tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_nearbyint_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rndne_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_nearbyint_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %nearbyint = call float @llvm.nearbyint.f32(float %a)
  %fneg = fneg float %nearbyint
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; fcanonicalize tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_canonicalize_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_mul_f32_e32 [[RESULT:v[0-9]+]], -1.0, [[A]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_fneg_canonicalize_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %trunc = call float @llvm.canonicalize.f32(float %a)
  %fneg = fneg float %trunc
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; vintrp tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_interp_p1_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], [[A]], -[[B]]
; GCN: v_interp_p1_f32{{(_e32)?}} v{{[0-9]+}}, [[MUL]]
; GCN: v_interp_p1_f32{{(_e32)?}} v{{[0-9]+}}, [[MUL]]
define amdgpu_kernel void @v_fneg_interp_p1_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  %intrp0 = call float @llvm.amdgcn.interp.p1(float %fneg, i32 0, i32 0, i32 0)
  %intrp1 = call float @llvm.amdgcn.interp.p1(float %fneg, i32 1, i32 0, i32 0)
  store volatile float %intrp0, float addrspace(1)* %out.gep
  store volatile float %intrp1, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_interp_p2_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], [[A]], -[[B]]
; GCN: v_interp_p2_f32{{(_e32)?}} v{{[0-9]+}}, [[MUL]]
; GCN: v_interp_p2_f32{{(_e32)?}} v{{[0-9]+}}, [[MUL]]
define amdgpu_kernel void @v_fneg_interp_p2_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  %intrp0 = call float @llvm.amdgcn.interp.p2(float 4.0, float %fneg, i32 0, i32 0, i32 0)
  %intrp1 = call float @llvm.amdgcn.interp.p2(float 4.0, float %fneg, i32 1, i32 0, i32 0)
  store volatile float %intrp0, float addrspace(1)* %out.gep
  store volatile float %intrp1, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; CopyToReg tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_copytoreg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN: v_mul_f32_e32 [[MUL0:v[0-9]+]], [[A]], [[B]]
; GCN: s_cbranch_scc0

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL0]]
; GCN: s_endpgm

; GCN: v_xor_b32_e32 [[XOR:v[0-9]+]], 0x80000000, [[MUL0]]
; GCN: v_mul_f32_e32 [[MUL1:v[0-9]+]], [[XOR]], [[C]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]

define amdgpu_kernel void @v_fneg_copytoreg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, i32 %d) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  %cmp0 = icmp eq i32 %d, 0
  br i1 %cmp0, label %if, label %endif

if:
  %mul1 = fmul float %fneg, %c
  store volatile float %mul1, float addrspace(1)* %out.gep
  br label %endif

endif:
  store volatile float %mul, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; inlineasm tests
; --------------------------------------------------------------------------------

; Can't fold into use, so should fold into source
; GCN-LABEL: {{^}}v_fneg_inlineasm_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], [[A]], -[[B]]
; GCN: ; use [[MUL]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_inlineasm_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, i32 %d) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  call void asm sideeffect "; use $0", "v"(float %fneg) #0
  store volatile float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; inlineasm tests
; --------------------------------------------------------------------------------

; Can't fold into use, so should fold into source
; GCN-LABEL: {{^}}v_fneg_inlineasm_multi_use_src_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], [[A]], [[B]]
; GCN: v_xor_b32_e32 [[NEG:v[0-9]+]], 0x80000000, [[MUL]]
; GCN: ; use [[NEG]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define amdgpu_kernel void @v_fneg_inlineasm_multi_use_src_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, i32 %d) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %mul = fmul float %a, %b
  %fneg = fneg float %mul
  call void asm sideeffect "; use $0", "v"(float %fneg) #0
  store volatile float %mul, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; code size regression tests
; --------------------------------------------------------------------------------

; There are multiple users of the fneg that must use a VOP3
; instruction, so there is no penalty
; GCN-LABEL: {{^}}multiuse_fneg_2_vop3_users_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN: v_fma_f32 [[FMA0:v[0-9]+]], -[[A]], [[B]], [[C]]
; GCN-NEXT: v_fma_f32 [[FMA1:v[0-9]+]], -[[A]], [[C]], 2.0

; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @multiuse_fneg_2_vop3_users_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep

  %fneg.a = fneg float %a
  %fma0 = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fma1 = call float @llvm.fma.f32(float %fneg.a, float %c, float 2.0)

  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; There are multiple users, but both require using a larger encoding
; for the modifier.

; GCN-LABEL: {{^}}multiuse_fneg_2_vop2_users_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN: v_mul_f32_e64 [[MUL0:v[0-9]+]], -[[A]], [[B]]
; GCN: v_mul_f32_e64 [[MUL1:v[0-9]+]], -[[A]], [[C]]
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @multiuse_fneg_2_vop2_users_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep

  %fneg.a = fneg float %a
  %mul0 = fmul float %fneg.a, %b
  %mul1 = fmul float %fneg.a, %c

  store volatile float %mul0, float addrspace(1)* %out
  store volatile float %mul1, float addrspace(1)* %out
  ret void
}

; One user is VOP3 so has no cost to folding the modifier, the other does.
; GCN-LABEL: {{^}}multiuse_fneg_vop2_vop3_users_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN: v_fma_f32 [[FMA0:v[0-9]+]], -[[A]], [[B]], 2.0
; GCN: v_mul_f32_e64 [[MUL1:v[0-9]+]], -[[A]], [[C]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @multiuse_fneg_vop2_vop3_users_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep

  %fneg.a = fneg float %a
  %fma0 = call float @llvm.fma.f32(float %fneg.a, float %b, float 2.0)
  %mul1 = fmul float %fneg.a, %c

  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %mul1, float addrspace(1)* %out
  ret void
}

; The use of the fneg requires a code size increase, but folding into
; the source does not

; GCN-LABEL: {{^}}free_fold_src_code_size_cost_use_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[D:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA0:v[0-9]+]], [[A]], [[B]], 2.0
; GCN-SAFE-DAG: v_mul_f32_e64 [[MUL1:v[0-9]+]], -[[FMA0]], [[C]]
; GCN-SAFE-DAG: v_mul_f32_e64 [[MUL2:v[0-9]+]], -[[FMA0]], [[D]]

; GCN-NSZ: v_fma_f32 [[FMA0:v[0-9]+]], [[A]], -[[B]], -2.0
; GCN-NSZ-DAG: v_mul_f32_e32 [[MUL1:v[0-9]+]], [[FMA0]], [[C]]
; GCN-NSZ-DAG: v_mul_f32_e32 [[MUL2:v[0-9]+]], [[FMA0]], [[D]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL2]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @free_fold_src_code_size_cost_use_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, float addrspace(1)* %d.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %d.gep = getelementptr inbounds float, float addrspace(1)* %d.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %d = load volatile float, float addrspace(1)* %d.gep

  %fma0 = call float @llvm.fma.f32(float %a, float %b, float 2.0)
  %fneg.fma0 = fneg float %fma0
  %mul1 = fmul float %fneg.fma0, %c
  %mul2 = fmul float %fneg.fma0, %d

  store volatile float %mul1, float addrspace(1)* %out
  store volatile float %mul2, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}free_fold_src_code_size_cost_use_f64:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: {{buffer|flat}}_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]]
; GCN: {{buffer|flat}}_load_dwordx2 [[C:v\[[0-9]+:[0-9]+\]]]
; GCN: {{buffer|flat}}_load_dwordx2 [[D:v\[[0-9]+:[0-9]+\]]]

; GCN: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], [[A]], [[B]], 2.0
; GCN-DAG: v_mul_f64 [[MUL0:v\[[0-9]+:[0-9]+\]]], -[[FMA0]], [[C]]
; GCN-DAG: v_mul_f64 [[MUL1:v\[[0-9]+:[0-9]+\]]], -[[FMA0]], [[D]]

; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[MUL0]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
define amdgpu_kernel void @free_fold_src_code_size_cost_use_f64(double addrspace(1)* %out, double addrspace(1)* %a.ptr, double addrspace(1)* %b.ptr, double addrspace(1)* %c.ptr, double addrspace(1)* %d.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds double, double addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds double, double addrspace(1)* %c.ptr, i64 %tid.ext
  %d.gep = getelementptr inbounds double, double addrspace(1)* %d.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %b = load volatile double, double addrspace(1)* %b.gep
  %c = load volatile double, double addrspace(1)* %c.gep
  %d = load volatile double, double addrspace(1)* %d.gep

  %fma0 = call double @llvm.fma.f64(double %a, double %b, double 2.0)
  %fneg.fma0 = fsub double -0.0, %fma0
  %mul1 = fmul double %fneg.fma0, %c
  %mul2 = fmul double %fneg.fma0, %d

  store volatile double %mul1, double addrspace(1)* %out
  store volatile double %mul2, double addrspace(1)* %out
  ret void
}

; %trunc.a has one fneg use, but it requires a code size increase and
; %the fneg can instead be folded for free into the fma.

; GCN-LABEL: {{^}}one_use_cost_to_fold_into_src_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN: v_trunc_f32_e32 [[TRUNC_A:v[0-9]+]], [[A]]
; GCN: v_fma_f32 [[FMA0:v[0-9]+]], -[[TRUNC_A]], [[B]], [[C]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA0]]
define amdgpu_kernel void @one_use_cost_to_fold_into_src_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, float addrspace(1)* %d.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %d.gep = getelementptr inbounds float, float addrspace(1)* %d.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %d = load volatile float, float addrspace(1)* %d.gep

  %trunc.a = call float @llvm.trunc.f32(float %a)
  %trunc.fneg.a = fneg float %trunc.a
  %fma0 = call float @llvm.fma.f32(float %trunc.fneg.a, float %b, float %c)
  store volatile float %fma0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}multi_use_cost_to_fold_into_src:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[D:v[0-9]+]]
; GCN: v_trunc_f32_e32 [[TRUNC_A:v[0-9]+]], [[A]]
; GCN-DAG: v_fma_f32 [[FMA0:v[0-9]+]], -[[TRUNC_A]], [[B]], [[C]]
; GCN-DAG: v_mul_f32_e32 [[MUL1:v[0-9]+]], [[TRUNC_A]], [[D]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FMA0]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MUL1]]
define amdgpu_kernel void @multi_use_cost_to_fold_into_src(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, float addrspace(1)* %d.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %c.gep = getelementptr inbounds float, float addrspace(1)* %c.ptr, i64 %tid.ext
  %d.gep = getelementptr inbounds float, float addrspace(1)* %d.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %c = load volatile float, float addrspace(1)* %c.gep
  %d = load volatile float, float addrspace(1)* %d.gep

  %trunc.a = call float @llvm.trunc.f32(float %a)
  %trunc.fneg.a = fneg float %trunc.a
  %fma0 = call float @llvm.fma.f32(float %trunc.fneg.a, float %b, float %c)
  %mul1 = fmul float %trunc.a, %d
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %mul1, float addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fma.f32(float, float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #1
declare float @llvm.sin.f32(float) #1
declare float @llvm.trunc.f32(float) #1
declare float @llvm.round.f32(float) #1
declare float @llvm.rint.f32(float) #1
declare float @llvm.nearbyint.f32(float) #1
declare float @llvm.canonicalize.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare half @llvm.minnum.f16(half, half) #1
declare double @llvm.minnum.f64(double, double) #1
declare double @llvm.fma.f64(double, double, double) #1

declare float @llvm.amdgcn.sin.f32(float) #1
declare float @llvm.amdgcn.rcp.f32(float) #1
declare float @llvm.amdgcn.rcp.legacy(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #0
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #0

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "unsafe-fp-math"="true" }
