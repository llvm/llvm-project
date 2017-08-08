; RUN: llc -march=amdgcn -mcpu=tahiti -start-after=sink -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -enable-no-signed-zeros-fp-math -march=amdgcn -mcpu=tahiti -start-after=sink -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NSZ -check-prefix=SI -check-prefix=FUNC %s

; --------------------------------------------------------------------------------
; fadd tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[A]], [[B]]
; GCN-NSZ-NEXT: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_add_store_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
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
; GCN-NSZ-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[ADD]]
; GCN: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_add_multi_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %add = fadd float %a, %b
  %fneg = fsub float -0.000000e+00, %add
  %use1 = fmul float %add, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e32
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000,

; GCN-NSZ: v_sub_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_add_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NSZ-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_add_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fsub float -0.000000e+00, %b
  %add = fadd float %a, %fneg.b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_sub_f32_e64 [[ADD:v[0-9]+]], -[[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ: v_add_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_add_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %add = fadd float %fneg.a, %fneg.b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE: v_bfrev_b32_e32 [[SIGNBIT:v[0-9]+]], 1{{$}}
; GCN-SAFE: v_xor_b32_e32 [[NEG_A:v[0-9]+]], [[A]], [[SIGNBIT]]
; GCN-SAFE: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], [[ADD]], [[SIGNBIT]]

; GCN-NSZ-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-NSZ-DAG: v_sub_f32_e32 [[NEG_ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NSZ-NEXT: buffer_store_dword [[NEG_A]]
define amdgpu_kernel void @v_fneg_add_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_add_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN-SAFE-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-SAFE-DAG: v_sub_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[ADD]]

; GCN-NSZ-DAG: v_sub_f32_e32 [[NEG_ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NSZ-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NSZ-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NSZ-NEXT: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_add_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %add = fadd float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %add
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fmul tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_store_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_mul_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[MUL0:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MUL0]]
; GCN-NEXT: buffer_store_dword [[MUL0]]
; GCN-NEXT: buffer_store_dword [[MUL1]]
define amdgpu_kernel void @v_fneg_mul_multi_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.000000e+00, %mul
  %use1 = fmul float %mul, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = fmul float %a, %fneg.b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = fmul float %fneg.a, %fneg.b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[NEG_A]]
define amdgpu_kernel void @v_fneg_mul_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_mul_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = fmul float %fneg.a, %b
  %fneg = fsub float -0.000000e+00, %mul
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fminnum tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_max_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -[[B]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_self_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float %a, float %a)
  %min.fneg = fsub float -0.0, %min
  store float %min.fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -4.0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_posk_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 4.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 [[RESULT:v[0-9]+]], -[[A]], 4.0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_negk_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float -4.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e32 [[RESULT:v[0-9]+]], 0, [[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_0_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float 0.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 [[RESULT:v[0-9]+]], -[[A]], 0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_neg0_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.minnum.f32(float -0.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_minnum_foldable_use_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], 0, [[A]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MIN]], [[B]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_0_minnum_foldable_use_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float 0.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  %mul = fmul float %fneg, %b
  store float %mul, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_minnum_multi_use_minnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_max_f32_e64 [[MAX0:v[0-9]+]], -[[A]], -[[B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MUL0]]
; GCN-NEXT: buffer_store_dword [[MAX0]]
; GCN-NEXT: buffer_store_dword [[MUL1]]
define amdgpu_kernel void @v_fneg_minnum_multi_use_minnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.minnum.f32(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %min
  %use1 = fmul float %min, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; --------------------------------------------------------------------------------
; fmaxnum tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_min_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -[[B]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_self_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_self_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.maxnum.f32(float %a, float %a)
  %min.fneg = fsub float -0.0, %min
  store float %min.fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_posk_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e64 [[RESULT:v[0-9]+]], -[[A]], -4.0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_posk_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.maxnum.f32(float 4.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_negk_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e64 [[RESULT:v[0-9]+]], -[[A]], 4.0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_negk_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %min = call float @llvm.maxnum.f32(float -4.0, float %a)
  %fneg = fsub float -0.000000e+00, %min
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_0_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float 0.0, float %a)
  %fneg = fsub float -0.000000e+00, %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_neg0_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_min_f32_e64 [[RESULT:v[0-9]+]], -[[A]], 0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_neg0_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %max = call float @llvm.maxnum.f32(float -0.0, float %a)
  %fneg = fsub float -0.000000e+00, %max
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_0_maxnum_foldable_use_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], 0, [[A]]
; GCN: v_mul_f32_e64 [[RESULT:v[0-9]+]], -[[MAX]], [[B]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_0_maxnum_foldable_use_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %max = call float @llvm.maxnum.f32(float 0.0, float %a)
  %fneg = fsub float -0.000000e+00, %max
  %mul = fmul float %fneg, %b
  store float %mul, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_maxnum_multi_use_maxnum_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_min_f32_e64 [[MAX0:v[0-9]+]], -[[A]], -[[B]]
; GCN-NEXT: v_mul_f32_e32 [[MUL1:v[0-9]+]], -4.0, [[MUL0]]
; GCN-NEXT: buffer_store_dword [[MAX0]]
; GCN-NEXT: buffer_store_dword [[MUL1]]
define amdgpu_kernel void @v_fneg_maxnum_multi_use_maxnum_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %min = call float @llvm.maxnum.f32(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %min
  %use1 = fmul float %min, 4.0
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
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
; GCN-NSZ-NEXT: buffer_store_dword [[RESULT]]
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
  %fneg = fsub float -0.000000e+00, %fma
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_store_use_fma_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN-DAG: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-DAG: v_xor_b32_e32 [[NEG_FMA:v[0-9]+]], 0x80000000, [[FMA]]
; GCN-NEXT: buffer_store_dword [[NEG_FMA]]
; GCN-NEXT: buffer_store_dword [[FMA]]
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
  %fneg = fsub float -0.000000e+00, %fma
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

; GCN-NEXT: buffer_store_dword [[NEG_FMA]]
; GCN-NEXT: buffer_store_dword [[MUL]]
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
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
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
  %fneg.a = fsub float -0.000000e+00, %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
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
  %fneg.b = fsub float -0.000000e+00, %b
  %fma = call float @llvm.fma.f32(float %a, float %fneg.b, float %c)
  %fneg = fsub float -0.000000e+00, %fma
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_fneg_fneg_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]], -[[A]], -[[B]], [[C]]
; GCN-SAFE: v_xor_b32_e32 v{{[[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
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
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %fma = call float @llvm.fma.f32(float %fneg.a, float %fneg.b, float %c)
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
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
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.c = fsub float -0.000000e+00, %c
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %fneg.c)
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
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
  %fneg.c = fsub float -0.000000e+00, %c
  %fma = call float @llvm.fma.f32(float %a, float %b, float %fneg.c)
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[FMA]]
; GCN-NSZ-NEXT: buffer_store_dword [[NEG_A]]
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
  %fneg.a = fsub float -0.000000e+00, %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fsub float -0.000000e+00, %fma
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fma_multi_use_fneg_x_y_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-SAFE: v_fma_f32 [[FMA:v[0-9]+]]
; GCN-SAFE: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, [[FMA]]

; GCN-NSZ-DAG: v_fma_f32 [[NEG_FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NSZ-NEXT: buffer_store_dword [[NEG_FMA]]
; GCN-NSZ-NEXT: buffer_store_dword [[MUL]]
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
  %fneg.a = fsub float -0.000000e+00, %a
  %fma = call float @llvm.fma.f32(float %fneg.a, float %b, float %c)
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN-NSZ-NEXT: buffer_store_dword [[RESULT]]
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
  %fneg = fsub float -0.000000e+00, %fma
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fmad_multi_use_fmad_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]

; GCN-SAFE: v_mac_f32_e32 [[C]], [[A]], [[B]]
; GCN-SAFE: v_xor_b32_e32 [[NEG_MAD:v[0-9]+]], 0x80000000, [[C]]
; GCN-SAFE-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[C]]

; GCN-NSZ: v_mad_f32 [[NEG_MAD:v[0-9]+]], -[[A]], [[B]], -[[C]]
; GCN-NSZ-NEXT: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[NEG_MAD]]

; GCN: buffer_store_dword [[NEG_MAD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
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
  %fneg = fsub float -0.000000e+00, %fma
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
; GCN: buffer_store_dwordx2 [[RESULT]]
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
; GCN: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fpext = fpext float %fneg.a to double
  %fneg = fsub double -0.000000e+00, %fpext
  store double %fneg, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_extend_store_use_fneg_f32_to_f64:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f64_f32_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[FNEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: buffer_store_dwordx2 [[RESULT]]
; GCN: buffer_store_dword [[FNEG_A]]
define amdgpu_kernel void @v_fneg_fp_extend_store_use_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
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
; GCN: buffer_store_dwordx2 v{{\[[0-9]+}}:[[FNEG_A]]{{\]}}
; GCN: buffer_store_dwordx2 v{{\[}}[[CVT_LO]]:[[CVT_HI]]{{\]}}
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
; GCN: buffer_store_dwordx2 v{{\[[0-9]+}}:[[FNEG_A]]{{\]}}
; GCN: buffer_store_dwordx2 [[MUL]]
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
  %fneg = fsub float -0.000000e+00, %fpext
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
  %fneg = fsub float -0.000000e+00, %fpext
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
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fpround = fptrunc double %a to float
  %fneg = fsub float -0.000000e+00, %fpround
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fsub float -0.000000e+00, %fpround
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_store_use_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 v{{\[}}[[A_LO:[0-9]+]]:[[A_HI:[0-9]+]]{{\]}}
; GCN-DAG: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], v{{\[}}[[A_LO]]:[[A_HI]]{{\]}}
; GCN-DAG: v_xor_b32_e32 v[[NEG_A_HI:[0-9]+]], 0x80000000, v[[A_HI]]
; GCN: buffer_store_dword [[RESULT]]
; GCN: buffer_store_dwordx2 v{{\[}}[[A_LO]]:[[NEG_A_HI]]{{\]}}
define amdgpu_kernel void @v_fneg_fp_round_store_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fsub float -0.000000e+00, %fpround
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile double %fneg.a, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_multi_use_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_cvt_f32_f64_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_mul_f64 [[USE1:v\[[0-9]+:[0-9]+\]]], -[[A]], s{{\[}}
; GCN: buffer_store_dword [[RESULT]]
; GCN: buffer_store_dwordx2 [[USE1]]
define amdgpu_kernel void @v_fneg_fp_round_multi_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr, double %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fneg.a = fsub double -0.000000e+00, %a
  %fpround = fptrunc double %fneg.a to float
  %fneg = fsub float -0.000000e+00, %fpround
  %use1 = fmul double %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile double %use1, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_cvt_f16_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_short [[RESULT]]
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
; GCN: buffer_store_short [[RESULT]]
define amdgpu_kernel void @v_fneg_fp_round_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fpround = fptrunc float %fneg.a to half
  %fneg = fsub half -0.000000e+00, %fpround
  store half %fneg, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_multi_use_fp_round_fneg_f64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_cvt_f32_f64_e32 [[CVT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG:v[0-9]+]], 0x80000000, [[CVT]]
; GCN: buffer_store_dword [[NEG]]
; GCN: buffer_store_dword [[CVT]]
define amdgpu_kernel void @v_fneg_multi_use_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds double, double addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile double, double addrspace(1)* %a.gep
  %fpround = fptrunc double %a to float
  %fneg = fsub float -0.000000e+00, %fpround
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %fpround, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fp_round_store_use_fneg_f32_to_f16:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: buffer_store_short [[RESULT]]
; GCN: buffer_store_dword [[NEG_A]]
define amdgpu_kernel void @v_fneg_fp_round_store_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
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
; GCN: buffer_store_short [[RESULT]]
; GCN: buffer_store_dword [[USE1]]
define amdgpu_kernel void @v_fneg_fp_round_multi_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
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
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_rcp_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %rcp = call float @llvm.amdgcn.rcp.f32(float %a)
  %fneg = fsub float -0.000000e+00, %rcp
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_rcp_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fsub float -0.000000e+00, %rcp
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_store_use_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN: buffer_store_dword [[RESULT]]
; GCN: buffer_store_dword [[NEG_A]]
define amdgpu_kernel void @v_fneg_rcp_store_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fsub float -0.000000e+00, %rcp
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %fneg.a, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_fneg_rcp_multi_use_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_rcp_f32_e32 [[RESULT:v[0-9]+]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN: buffer_store_dword [[RESULT]]
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_rcp_multi_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %rcp = call float @llvm.amdgcn.rcp.f32(float %fneg.a)
  %fneg = fsub float -0.000000e+00, %rcp
  %use1 = fmul float %fneg.a, %c
  store volatile float %fneg, float addrspace(1)* %out.gep
  store volatile float %use1, float addrspace(1)* undef
  ret void
}

; --------------------------------------------------------------------------------
; rcp_legacy tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_rcp_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rcp_legacy_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_rcp_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %rcp = call float @llvm.amdgcn.rcp.legacy(float %a)
  %fneg = fsub float -0.000000e+00, %rcp
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; fmul_legacy tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[RESULT:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_store_use_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_store_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_multi_use_mul_legacy_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: v_mul_legacy_f32_e64 [[MUL:v[0-9]+]], -[[ADD]], 4.0
; GCN-NEXT: buffer_store_dword [[ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_mul_legacy_multi_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
  %use1 = call float @llvm.amdgcn.fmul.legacy(float %mul, float 4.0)
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %use1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_x_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %fneg.b)
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_fneg_fneg_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_mul_legacy_f32_e64 [[ADD:v[0-9]+]], [[A]], -[[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_fneg_mul_legacy_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %fneg.b = fsub float -0.000000e+00, %b
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %fneg.b)
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_store_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[NEG_A]]
define amdgpu_kernel void @v_fneg_mul_legacy_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
  store volatile float %fneg, float addrspace(1)* %out
  store volatile float %fneg.a, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fneg_mul_legacy_multi_use_fneg_x_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[A]], [[B]]
; GCN-DAG: v_mul_legacy_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @v_fneg_mul_legacy_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %fneg.a = fsub float -0.000000e+00, %a
  %mul = call float @llvm.amdgcn.fmul.legacy(float %fneg.a, float %b)
  %fneg = fsub float -0.000000e+00, %mul
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
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %sin = call float @llvm.sin.f32(float %a)
  %fneg = fsub float -0.000000e+00, %sin
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_fneg_amdgcn_sin_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_sin_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_amdgcn_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %sin = call float @llvm.amdgcn.sin.f32(float %a)
  %fneg = fsub float -0.0, %sin
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; ftrunc tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_trunc_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_trunc_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_trunc_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %trunc = call float @llvm.trunc.f32(float %a)
  %fneg = fsub float -0.0, %trunc
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
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_round_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %round = call float @llvm.round.f32(float %a)
  %fneg = fsub float -0.0, %round
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; rint tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_rint_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rndne_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_rint_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %rint = call float @llvm.rint.f32(float %a)
  %fneg = fsub float -0.0, %rint
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

; --------------------------------------------------------------------------------
; nearbyint tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_nearbyint_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: v_rndne_f32_e64 [[RESULT:v[0-9]+]], -[[A]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @v_fneg_nearbyint_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %nearbyint = call float @llvm.nearbyint.f32(float %a)
  %fneg = fsub float -0.0, %nearbyint
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
; GCN: v_interp_p1_f32 v{{[0-9]+}}, [[MUL]]
; GCN: v_interp_p1_f32 v{{[0-9]+}}, [[MUL]]
define amdgpu_kernel void @v_fneg_interp_p1_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.0, %mul
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
; GCN: v_interp_p2_f32 v{{[0-9]+}}, [[MUL]]
; GCN: v_interp_p2_f32 v{{[0-9]+}}, [[MUL]]
define amdgpu_kernel void @v_fneg_interp_p2_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds float, float addrspace(1)* %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %b = load volatile float, float addrspace(1)* %b.gep
  %mul = fmul float %a, %b
  %fneg = fsub float -0.0, %mul
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
; GCN: s_cbranch_scc1

; GCN: v_xor_b32_e32 [[XOR:v[0-9]+]], 0x80000000, [[MUL0]]
; GCN: v_mul_f32_e32 [[MUL1:v[0-9]+]], [[XOR]], [[C]]
; GCN: buffer_store_dword [[MUL1]]

; GCN: buffer_store_dword [[MUL0]]
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
  %fneg = fsub float -0.0, %mul
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
; GCN: buffer_store_dword [[MUL]]
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
  %fneg = fsub float -0.0, %mul
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
; GCN: buffer_store_dword [[MUL]]
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
  %fneg = fsub float -0.0, %mul
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
; GCN-NEXT:	buffer_store_dword [[FMA0]]
; GCN-NEXT:	buffer_store_dword [[FMA1]]
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

  %fneg.a = fsub float -0.0, %a
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
; GCN-NEXT:	buffer_store_dword [[MUL0]]
; GCN-NEXT:	buffer_store_dword [[MUL1]]
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

  %fneg.a = fsub float -0.0, %a
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

; GCN:	buffer_store_dword [[FMA0]]
; GCN-NEXT:	buffer_store_dword [[MUL1]]
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

  %fneg.a = fsub float -0.0, %a
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

; GCN: buffer_store_dword [[MUL1]]
; GCN-NEXT:	buffer_store_dword [[MUL2]]
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
  %fneg.fma0 = fsub float -0.0, %fma0
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

; GCN: buffer_store_dwordx2 [[MUL0]]
; GCN: buffer_store_dwordx2 [[MUL1]]
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
; GCN: buffer_store_dword [[FMA0]]
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
  %trunc.fneg.a = fsub float -0.0, %trunc.a
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
; GCN: buffer_store_dword [[FMA0]]
; GCN: buffer_store_dword [[MUL1]]
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
  %trunc.fneg.a = fsub float -0.0, %trunc.a
  %fma0 = call float @llvm.fma.f32(float %trunc.fneg.a, float %b, float %c)
  %mul1 = fmul float %trunc.a, %d
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %mul1, float addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fma.f32(float, float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.sin.f32(float) #1
declare float @llvm.trunc.f32(float) #1
declare float @llvm.round.f32(float) #1
declare float @llvm.rint.f32(float) #1
declare float @llvm.nearbyint.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1

declare double @llvm.fma.f64(double, double, double) #1

declare float @llvm.amdgcn.sin.f32(float) #1
declare float @llvm.amdgcn.rcp.f32(float) #1
declare float @llvm.amdgcn.rcp.legacy(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #0
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
