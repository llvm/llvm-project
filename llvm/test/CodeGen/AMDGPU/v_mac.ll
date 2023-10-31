; RUN:  llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mattr=+mad-mac-f32-insts -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck --check-prefixes=SI,GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -denormal-fp-math=preserve-sign -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=VI-FLUSH,GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -denormal-fp-math=ieee -denormal-fp-math-f32=preserve-sign -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}mac_vvv:
; GCN: buffer_load_dword [[A:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 glc{{$}}
; GCN: buffer_load_dword [[B:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:4
; GCN: buffer_load_dword [[C:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:8
; GCN: v_mac_f32_e32 [[C]], [[A]], [[B]]
; GCN: buffer_store_dword [[C]]
define amdgpu_kernel void @mac_vvv(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load volatile float, ptr addrspace(1) %in
  %b = load volatile float, ptr addrspace(1) %b_ptr
  %c = load volatile float, ptr addrspace(1) %c_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mad_inline_sgpr_inline:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]}}, s{{[0-9]+}}, 0.5, 0.5
define amdgpu_kernel void @mad_inline_sgpr_inline(ptr addrspace(1) %out, float %in) #0 {
entry:
  %tmp0 = fmul float 0.5, %in
  %tmp1 = fadd float %tmp0, 0.5
  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mad_vvs:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @mad_vvs(ptr addrspace(1) %out, ptr addrspace(1) %in, float %c) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mac_ssv:
; GCN: v_mac_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @mac_ssv(ptr addrspace(1) %out, ptr addrspace(1) %in, float %a) #0 {
entry:
  %c = load float, ptr addrspace(1) %in

  %tmp0 = fmul float %a, %a
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mac_mad_same_add:
; GCN: v_mad_f32 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD:v[0-9]+]]
; GCN: v_mac_f32_e32 [[ADD]], v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @mac_mad_same_add(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2
  %d_ptr = getelementptr float, ptr addrspace(1) %in, i32 3
  %e_ptr = getelementptr float, ptr addrspace(1) %in, i32 4

  %a = load volatile float, ptr addrspace(1) %in
  %b = load volatile float, ptr addrspace(1) %b_ptr
  %c = load volatile float, ptr addrspace(1) %c_ptr
  %d = load volatile float, ptr addrspace(1) %d_ptr
  %e = load volatile float, ptr addrspace(1) %e_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c

  %tmp2 = fmul float %d, %e
  %tmp3 = fadd float %tmp2, %c

  %out1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %tmp1, ptr addrspace(1) %out
  store float %tmp3, ptr addrspace(1) %out1
  ret void
}

; There is no advantage to using v_mac when one of the operands is negated
; and v_mad accepts more operand types.

; GCN-LABEL: {{^}}mad_neg_src0:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define amdgpu_kernel void @mad_neg_src0(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_a = fneg float %a
  %tmp0 = fmul float %neg_a, %b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}nsz_mad_sub0_src0:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define amdgpu_kernel void @nsz_mad_sub0_src0(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_a = fsub float 0.0, %a
  %tmp0 = fmul float %neg_a, %b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}safe_mad_sub0_src0:
; GCN: v_sub_f32_e32 [[SUB0:v[0-9]+]], 0,
; GCN: v_ma{{[cd]}}_f32{{[_e32]*}} v{{[0-9]+}}, [[SUB0]], v{{[0-9]+}}
define amdgpu_kernel void @safe_mad_sub0_src0(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_a = fsub float 0.0, %a
  %tmp0 = fmul float %neg_a, %b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mad_neg_src1:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define amdgpu_kernel void @mad_neg_src1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_b = fneg float %b
  %tmp0 = fmul float %a, %neg_b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}nsz_mad_sub0_src1:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define amdgpu_kernel void @nsz_mad_sub0_src1(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_b = fsub float 0.0, %b
  %tmp0 = fmul float %a, %neg_b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mad_neg_src2:
; GCN-NOT: v_mac
; GCN: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
define amdgpu_kernel void @mad_neg_src2(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %b_ptr = getelementptr float, ptr addrspace(1) %in, i32 1
  %c_ptr = getelementptr float, ptr addrspace(1) %in, i32 2

  %a = load float, ptr addrspace(1) %in
  %b = load float, ptr addrspace(1) %b_ptr
  %c = load float, ptr addrspace(1) %c_ptr

  %neg_c = fneg float %c
  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %neg_c

  store float %tmp1, ptr addrspace(1) %out
  ret void
}

; Without special casing the inline constant check for v_mac_f32's
; src2, this fails to fold the 1.0 into a mad.

; GCN-LABEL: {{^}}fold_inline_imm_into_mac_src2_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; GCN: v_add_f32_e32 [[TMP2:v[0-9]+]], [[A]], [[A]]
; GCN: v_mad_f32 v{{[0-9]+}}, [[TMP2]], -4.0, 1.0
define amdgpu_kernel void @fold_inline_imm_into_mac_src2_f32(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) #3 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds float, ptr addrspace(1) %a, i64 %tid.ext
  %gep.b = getelementptr inbounds float, ptr addrspace(1) %b, i64 %tid.ext
  %gep.out = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %tmp = load volatile float, ptr addrspace(1) %gep.a
  %tmp1 = load volatile float, ptr addrspace(1) %gep.b
  %tmp2 = fadd float %tmp, %tmp
  %tmp3 = fmul float %tmp2, 4.0
  %tmp4 = fsub float 1.0, %tmp3
  %tmp5 = fadd float %tmp4, %tmp1
  %tmp6 = fadd float %tmp1, %tmp1
  %tmp7 = fmul float %tmp6, %tmp
  %tmp8 = fsub float 1.0, %tmp7
  %tmp9 = fmul float %tmp8, 8.0
  %tmp10 = fadd float %tmp5, %tmp9
  store float %tmp10, ptr addrspace(1) %gep.out
  ret void
}

; GCN-LABEL: {{^}}fold_inline_imm_into_mac_src2_f16:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort [[B:v[0-9]+]]

; SI-DAG: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], [[A]]
; SI-DAG: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], [[B]]

; SI: v_add_f32_e32 [[TMP2:v[0-9]+]], [[CVT_A]], [[CVT_A]]
; SI: v_mad_f32 v{{[0-9]+}}, [[TMP2]], -4.0, 1.0
; SI: v_madmk_f32 v{{[0-9]+}}, v{{[0-9]+}}, 0x41000000, v{{[0-9]+}}

; VI-FLUSH: v_add_f16_e32 [[TMP2:v[0-9]+]], [[A]], [[A]]
; VI-FLUSH: v_mad_f16 v{{[0-9]+}}, [[TMP2]], -4.0, 1.0
define amdgpu_kernel void @fold_inline_imm_into_mac_src2_f16(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) #3 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds half, ptr addrspace(1) %a, i64 %tid.ext
  %gep.b = getelementptr inbounds half, ptr addrspace(1) %b, i64 %tid.ext
  %gep.out = getelementptr inbounds half, ptr addrspace(1) %out, i64 %tid.ext
  %tmp = load volatile half, ptr addrspace(1) %gep.a
  %tmp1 = load volatile half, ptr addrspace(1) %gep.b
  %tmp2 = fadd half %tmp, %tmp
  %tmp3 = fmul half %tmp2, 4.0
  %tmp4 = fsub half 1.0, %tmp3
  %tmp5 = fadd half %tmp4, %tmp1
  %tmp6 = fadd half %tmp1, %tmp1
  %tmp7 = fmul half %tmp6, %tmp
  %tmp8 = fsub half 1.0, %tmp7
  %tmp9 = fmul half %tmp8, 8.0
  %tmp10 = fadd half %tmp5, %tmp9
  store half %tmp10, ptr addrspace(1) %gep.out
  ret void
}

; Need to assume denormal handling is needed for dynamic denormal mode
; GCN-LABEL: {{^}}v_mac_f32_dynamic:
; GCN: v_mul_f32
; GCN: v_add_f32
define float @v_mac_f32_dynamic(float %a, float %b, float %c) "denormal-fp-math-f32"="dynamic,dynamic" {
  %mul = fmul float %a, %b
  %mad = fadd float %mul, %c
  ret float %mad
}

; GCN-LABEL: {{^}}v_mac_f32_dynamic_daz:
; GCN: v_mul_f32
; GCN: v_add_f32
define float @v_mac_f32_dynamic_daz(float %a, float %b, float %c) "denormal-fp-math-f32"="preserve-sign,dynamic" {
  %mul = fmul float %a, %b
  %mad = fadd float %mul, %c
  ret float %mad
}

; GCN-LABEL: {{^}}v_mac_f32_dynamic_ftz:
; GCN: v_mul_f32
; GCN: v_add_f32
define float @v_mac_f32_dynamic_ftz(float %a, float %b, float %c) "denormal-fp-math-f32"="dynamic,preserve-sign" {
  %mul = fmul float %a, %b
  %mad = fadd float %mul, %c
  ret float %mad
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind "no-signed-zeros-fp-math"="false" }
attributes #1 = { nounwind "no-signed-zeros-fp-math"="true" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
