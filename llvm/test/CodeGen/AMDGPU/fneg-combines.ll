; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s

; --------------------------------------------------------------------------------
; fadd tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_add_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define void @v_fneg_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_store_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_ADD:v[0-9]+]], 0x80000000, [[ADD]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_add_multi_use_add_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_subrev_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_subrev_f32_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_add_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_subrev_f32_e32 [[NEG_ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_A]]
define void @v_fneg_add_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_subrev_f32_e32 [[NEG_ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_ADD]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_add_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
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
define void @v_fneg_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[ADD]]
define void @v_fneg_mul_store_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL:v[0-9]+]], 0x80000000, [[ADD]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_multi_use_mul_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_mul_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
define void @v_fneg_mul_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[NEG_A]]
define void @v_fneg_mul_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_f32_e32 [[NEG_MUL:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_MUL]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
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
; fma tests
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}v_fneg_fma_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[C:v[0-9]+]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define void @v_fneg_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
define void @v_fneg_fma_store_use_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN-DAG: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-DAG: v_xor_b32_e32 [[NEG_FMA:v[0-9]+]], 0x80000000, [[FMA]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[FMA]]
; GCN-NEXT: buffer_store_dword [[NEG_FMA]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_fma_multi_use_fma_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
define void @v_fneg_fma_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
define void @v_fneg_fma_x_fneg_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
define void @v_fneg_fma_fneg_fneg_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], [[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
define void @v_fneg_fma_fneg_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[A]], -[[B]], [[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
define void @v_fneg_fma_x_y_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN-DAG: v_xor_b32_e32 [[NEG_A:v[0-9]+]], 0x80000000, [[A]]
; GCN-DAG: v_fma_f32 [[FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[FMA]]
; GCN-NEXT: buffer_store_dword [[NEG_A]]
define void @v_fneg_fma_store_use_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN-DAG: v_mul_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-DAG: v_fma_f32 [[NEG_FMA:v[0-9]+]], [[A]], [[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[NEG_FMA]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_fma_multi_use_fneg_x_y_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr, float %d) #0 {
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
; GCN: v_mad_f32 [[RESULT:v[0-9]+]], [[A]], -[[B]], -[[C]]
; GCN-NEXT: buffer_store_dword [[RESULT]]
define void @v_fneg_fmad_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
; GCN-DAG: v_mac_f32_e32 [[C]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_C:v[0-9]+]], 0x80000000, [[C]]
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 4.0, [[C]]
; GCN-NEXT: buffer_store_dword [[NEG_C]]
; GCN-NEXT: buffer_store_dword [[MUL]]
define void @v_fneg_fmad_multi_use_fmad_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float addrspace(1)* %c.ptr) #0 {
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
define void @v_fneg_fp_extend_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_extend_store_use_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_multi_use_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_multi_foldable_use_fp_extend_fneg_f32_to_f64(double addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_multi_use_fp_extend_fneg_f16_to_f32(float addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_multi_foldable_use_fp_extend_fneg_f16_to_f32(float addrspace(1)* %out, half addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
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
; GCN-DAG: v_mov_b32_e32 v[[NEG_A_LO:[0-9]+]], v[[A_LO]]
; GCN: buffer_store_dword [[RESULT]]
; GCN: buffer_store_dwordx2 v{{\[}}[[NEG_A_LO]]:[[NEG_A_HI]]{{\]}}
define void @v_fneg_fp_round_store_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_multi_use_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr, double %c) #0 {
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
define void @v_fneg_fp_round_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_multi_use_fp_round_fneg_f64_to_f32(float addrspace(1)* %out, double addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_store_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_fp_round_multi_use_fneg_f32_to_f16(half addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
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
define void @v_fneg_rcp_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_rcp_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_rcp_store_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_rcp_multi_use_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float %c) #0 {
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
define void @v_fneg_rcp_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], 0x80000000, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[ADD]]
define void @v_fneg_mul_legacy_store_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_xor_b32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], 0x80000000, [[ADD]]
; GCN: v_mul_legacy_f32_e32 [[MUL:v[0-9]+]], 4.0, [[ADD]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_legacy_multi_use_mul_legacy_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_legacy_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN: v_mul_legacy_f32_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[ADD]]
define void @v_fneg_mul_legacy_x_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
define void @v_fneg_mul_legacy_fneg_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[B]], [[A]]
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[NEG_A]]
define void @v_fneg_mul_legacy_store_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr) #0 {
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
; GCN-DAG: v_mul_legacy_f32_e32 [[NEG_MUL_LEGACY:v[0-9]+]], [[B]], [[A]]
; GCN-DAG: v_mul_legacy_f32_e64 [[MUL:v[0-9]+]], -[[A]], s{{[0-9]+}}
; GCN-NEXT: buffer_store_dword [[NEG_MUL_LEGACY]]
; GCN: buffer_store_dword [[MUL]]
define void @v_fneg_mul_legacy_multi_use_fneg_x_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr, float addrspace(1)* %b.ptr, float %c) #0 {
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
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e22f983
; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], [[K]], -[[A]]
; GCN: v_fract_f32_e32 [[FRACT:v[0-9]+]], [[MUL]]
; GCN: v_sin_f32_e32 [[RESULT:v[0-9]+]], [[FRACT]]
; GCN: buffer_store_dword [[RESULT]]
define void @v_fneg_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
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
define void @v_fneg_amdgcn_sin_f32(float addrspace(1)* %out, float addrspace(1)* %a.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds float, float addrspace(1)* %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %a = load volatile float, float addrspace(1)* %a.gep
  %sin = call float @llvm.amdgcn.sin.f32(float %a)
  %fneg = fsub float -0.000000e+00, %sin
  store float %fneg, float addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fma.f32(float, float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.sin.f32(float) #1

declare float @llvm.amdgcn.sin.f32(float) #1
declare float @llvm.amdgcn.rcp.f32(float) #1
declare float @llvm.amdgcn.rcp.legacy(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
