; RUN: llc -march=amdgcn -mattr=+fast-fmaf,-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-FLUSH %s
; RUN: llc -march=amdgcn -mattr=-fast-fmaf,-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-FLUSH %s

; RUN: llc -march=amdgcn -mattr=+fast-fmaf,+fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-FASTFMA %s
; RUN: llc -march=amdgcn -mattr=-fast-fmaf,+fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-SLOWFMA %s

; FIXME: This should also fold when fma is actually fast if an FMA
; exists in the original program.

; (fadd (fma x, y, (fmul u, v), z) -> (fma x, y (fma u, v, z))

; GCN-LABEL: {{^}}fast_add_fmuladd_fmul:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-FLUSH: v_mac_f32_e32 [[Z]], [[U]], [[V]]
; GCN-FLUSH-NEXT: v_mac_f32_e32 [[Z]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: buffer_store_dword [[Z]]

; GCN-FASTFMA: v_fma_f32 [[FMA0:v[0-9]+]], [[U]], [[V]], [[Z]]
; GCN-FASTFMA: v_fma_f32 [[FMA1:v[0-9]+]], [[X]], [[Y]], [[FMA0]]
; GCN-FASTFMA: buffer_store_dword [[FMA1]]

; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
define amdgpu_kernel void @fast_add_fmuladd_fmul() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fadd fast float %fma, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fmul:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-FLUSH: v_mad_f32 [[TMP:v[0-9]]], [[U]], [[V]], -[[Z]]
; GCN-FLUSH-NEXT: v_mac_f32_e32 [[TMP]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: buffer_store_dword [[Z]]

; GCN-FASTFMA: v_fma_f32 [[FMA0:v[0-9]+]], [[U]], [[V]], -[[Z]]
; GCN-FASTFMA: v_fma_f32 [[FMA1:v[0-9]+]], [[X]], [[Y]], [[FMA0]]
; GCN-FASTFMA: buffer_store_dword [[FMA1]]
define amdgpu_kernel void @fast_sub_fmuladd_fmul() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %fma, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_add_fmuladd_fmul_multi_use_mul:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-FLUSH-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]
; GCN-FLUSH-DAG: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH: v_add_f32_e32 v{{[0-9]+}}, [[U]], [[Z]]

; GCN-FASTFMA: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]
; GCN-FASTFMA: v_fma_f32 [[FMA1:v[0-9]+]], [[X]], [[Y]], [[MUL]]
; GCN-FASTFMA: v_add_f32_e32 v{{[0-9]+}}, [[FMA1]], [[Z]]

; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
define amdgpu_kernel void @fast_add_fmuladd_fmul_multi_use_mul() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  store volatile float %mul.u.v, float addrspace(1)* undef
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fadd fast float %fma, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_add_fmuladd_fmul_multi_use_mul_commute:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-FLUSH-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]
; GCN-FLUSH-DAG: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[U]]

; GCN-FASTFMA: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]
; GCN-FASTFMA: v_fma_f32 [[FMA1:v[0-9]+]], [[X]], [[Y]], [[MUL]]
; GCN-FASTFMA: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[FMA1]]

; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
define amdgpu_kernel void @fast_add_fmuladd_fmul_multi_use_mul_commute() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  store volatile float %mul.u.v, float addrspace(1)* undef
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fadd fast float %z, %fma
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_add_fmuladd_fmul_multi_use_fmuladd:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
define amdgpu_kernel void @fast_add_fmuladd_fmul_multi_use_fmuladd() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  store volatile float %fma, float addrspace(1)* undef
  %add = fadd fast float %fma, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_add_fmuladd_fmul_multi_use_fmuladd_commute:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_mul_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_add_f32_e32
define amdgpu_kernel void @fast_add_fmuladd_fmul_multi_use_fmuladd_commute() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  store volatile float %fma, float addrspace(1)* undef
  %add = fadd fast float %z, %fma
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fmul_multi_use_mul:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]

; GCN-FLUSH: v_mad_f32 [[MAD:v[0-9]+]], [[X]], [[Y]], [[MUL]]
; GCN-FLUSH: v_sub_f32_e32 [[SUB:v[0-9]+]], [[MAD]], [[Z]]

; GCN-FASTFMA: v_fma_f32 [[MAD:v[0-9]+]], [[X]], [[Y]], [[MUL]]
; GCN-FASTFMA: v_sub_f32_e32 [[SUB:v[0-9]+]], [[MAD]], [[Z]]

; GCN-SLOWFMA-DAG: v_mul_f32_e32 v{{[0-9]+}}, [[X]], [[Y]]
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_sub_f32_e32 [[MAD:v[0-9]+]]

; GCN: buffer_store_dword [[MUL]]
; GCN: buffer_store_dword [[MAD]]
define amdgpu_kernel void @fast_sub_fmuladd_fmul_multi_use_mul() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %fma, %z
  store volatile float %mul.u.v, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fmul_multi_use_fmuladd_lhs:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]

; GCN-FLUSH-NEXT: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]],  [[MUL]], [[Z]]
; GCN-FLUSH-NEXT: buffer_store_dword [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[SUB]]

; GCN-FASTFMA-NEXT: v_fma_f32 [[FMA:v[0-9]+]], [[X]], [[Y]], [[U]]
; GCN-FASTFMA-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]], [[FMA]], [[Z]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[SUB]]

; GCN-SLOWFMA-DAG: v_mul_f32_e32 v{{[0-9]+}}, [[X]], [[Y]]
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_sub_f32_e32
define amdgpu_kernel void @fast_sub_fmuladd_fmul_multi_use_fmuladd_lhs() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %fma, %z
  store volatile float %fma, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fmul_multi_use_fmuladd_rhs:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[U:v[0-9]+]]
; GCN: buffer_load_dword [[V:v[0-9]+]]

; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[U]], [[V]]

; GCN-FLUSH-NEXT: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]],  [[Z]], [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[SUB]]

; GCN-FASTFMA-NEXT: v_fma_f32 [[FMA:v[0-9]+]], [[X]], [[Y]], [[U]]
; GCN-FASTFMA-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]], [[Z]], [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[SUB]]

; GCN-SLOWFMA-DAG: v_mul_f32_e32 v{{[0-9]+}}, [[X]], [[Y]]
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_sub_f32_e32
define amdgpu_kernel void @fast_sub_fmuladd_fmul_multi_use_fmuladd_rhs() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile float, float addrspace(1)* undef
  %v = load volatile float, float addrspace(1)* undef
  %mul.u.v = fmul fast float %u, %v
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %z, %fma
  store volatile float %fma, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fpext_fmul_multi_use_fmuladd_lhs:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_ushort [[U:v[0-9]+]]
; GCN: buffer_load_ushort [[V:v[0-9]+]]

; GCN-DAG: v_cvt_f32_f16_e32 [[UFLOAT:v[0-9]+]], [[U]]
; GCN-DAG: v_cvt_f32_f16_e32 [[VFLOAT:v[0-9]+]], [[V]]
; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[UFLOAT]], [[VFLOAT]]

; GCN-FLUSH-NEXT: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]],  [[MUL]], [[Z]]
; GCN-FLUSH-NEXT: buffer_store_dword [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[SUB]]

; GCN-FASTFMA-NEXT: v_fma_f32 [[FMA:v[0-9]+]], [[X]], [[Y]], [[UFLOAT]]
; GCN-FASTFMA-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]], [[FMA]], [[Z]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[SUB]]

; GCN-SLOWFMA-DAG: v_mul_f32_e32 v{{[0-9]+}}, [[X]], [[Y]]
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_sub_f32_e32
define amdgpu_kernel void @fast_sub_fmuladd_fpext_fmul_multi_use_fmuladd_lhs() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile half, half addrspace(1)* undef
  %v = load volatile half, half addrspace(1)* undef
  %mul.u.v.half = fmul fast half %u, %v
  %mul.u.v = fpext half %mul.u.v.half to float
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %fma, %z
  store volatile float %fma, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fast_sub_fmuladd_fpext_fmul_multi_use_fmuladd_rhs:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_ushort [[U:v[0-9]+]]
; GCN: buffer_load_ushort [[V:v[0-9]+]]

; GCN-DAG: v_cvt_f32_f16_e32 [[UFLOAT:v[0-9]+]], [[U]]
; GCN-DAG: v_cvt_f32_f16_e32 [[VFLOAT:v[0-9]+]], [[V]]
; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[UFLOAT]], [[VFLOAT]]

; GCN-FLUSH-NEXT: v_mac_f32_e32 [[MUL]], [[X]], [[Y]]
; GCN-FLUSH-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]],  [[Z]], [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[MUL]]
; GCN-FLUSH-NEXT: buffer_store_dword [[SUB]]

; GCN-FASTFMA-NEXT: v_fma_f32 [[FMA:v[0-9]+]], [[X]], [[Y]], [[UFLOAT]]
; GCN-FASTFMA-NEXT: v_sub_f32_e32 [[SUB:v[0-9]+]], [[Z]], [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[FMA]]
; GCN-FASTFMA-NEXT: buffer_store_dword [[SUB]]

; GCN-SLOWFMA-DAG: v_mul_f32_e32 v{{[0-9]+}}, [[X]], [[Y]]
; GCN-SLOWFMA: v_add_f32_e32
; GCN-SLOWFMA: v_sub_f32_e32
define amdgpu_kernel void @fast_sub_fmuladd_fpext_fmul_multi_use_fmuladd_rhs() #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %u = load volatile half, half addrspace(1)* undef
  %v = load volatile half, half addrspace(1)* undef
  %mul.u.v.half = fmul fast half %u, %v
  %mul.u.v = fpext half %mul.u.v.half to float
  %fma = call fast float @llvm.fmuladd.f32(float %x, float %y, float %mul.u.v)
  %add = fsub fast float %z, %fma
  store volatile float %fma, float addrspace(1)* undef
  store volatile float %add, float addrspace(1)* undef
  ret void
}

declare float @llvm.fma.f32(float, float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
