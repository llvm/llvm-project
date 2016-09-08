; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure (fmul (fadd x, x), c) -> (fmul x, (fmul 2.0, c)) doesn't
; make add an instruction if the fadd has more than one use.

declare float @llvm.fabs.f32(float) #1

; GCN-LABEL: {{^}}multiple_fadd_use_test:
; GCN: v_max_legacy_f32_e64 [[A16:v[0-9]+]],
; GCN: v_add_f32_e32 [[A17:v[0-9]+]], [[A16]], [[A16]]
; GCN: v_mul_f32_e32 [[A18:v[0-9]+]], [[A17]], [[A17]]
; GCN: v_mad_f32 [[A20:v[0-9]+]], -[[A18]], [[A17]], 1.0
; GCN: buffer_store_dword [[A20]]
define void @multiple_fadd_use_test(float addrspace(1)* %out, float %x, float %y, float %z) #0 {
  %a11 = fadd fast float %y, -1.0
  %a12 = call float @llvm.fabs.f32(float %a11)
  %a13 = fadd fast float %x, -1.0
  %a14 = call float @llvm.fabs.f32(float %a13)
  %a15 = fcmp ogt float %a12, %a14
  %a16 = select i1 %a15, float %a12, float %a14
  %a17 = fmul fast float %a16, 2.0
  %a18 = fmul fast float %a17, %a17
  %a19 = fmul fast float %a18, %a17
  %a20 = fsub fast float 1.0, %a19
  store float %a20, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}multiple_use_fadd_fmac
; GCN-DAG: v_add_f32_e64 [[MUL2:v[0-9]+]], [[X:s[0-9]+]], s{{[0-9]+}}
; GCN-DAG: v_mac_f32_e64 [[MAD:v[0-9]+]], 2.0, [[X]]
; GCN-DAG: buffer_store_dword [[MUL2]]
; GCN-DAG: buffer_store_dword [[MAD]]
; GCN: s_endpgm
define void @multiple_use_fadd_fmac(float addrspace(1)* %out, float %x, float %y) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %mul2 = fmul fast float %x, 2.0
  %mad = fadd fast float %mul2, %y
  store volatile float %mul2, float addrspace(1)* %out
  store volatile float %mad, float addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}multiple_use_fadd_fmad:
; GCN-DAG: v_add_f32_e64 [[MUL2:v[0-9]+]], |[[X:s[0-9]+]]|, |s{{[0-9]+}}|
; GCN-DAG: v_mad_f32 [[MAD:v[0-9]+]], 2.0, |[[X]]|, v{{[0-9]+}}
; GCN-DAG: buffer_store_dword [[MUL2]]
; GCN-DAG: buffer_store_dword [[MAD]]
; GCN: s_endpgm
define void @multiple_use_fadd_fmad(float addrspace(1)* %out, float %x, float %y) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %x.abs = call float @llvm.fabs.f32(float %x)
  %mul2 = fmul fast float %x.abs, 2.0
  %mad = fadd fast float %mul2, %y
  store volatile float %mul2, float addrspace(1)* %out
  store volatile float %mad, float addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}multiple_use_fadd_multi_fmad:
; GCN: v_mad_f32 {{v[0-9]+}}, 2.0, |[[X:s[0-9]+]]|, v{{[0-9]+}}
; GCN: v_mad_f32 {{v[0-9]+}}, 2.0, |[[X]]|, v{{[0-9]+}}
define void @multiple_use_fadd_multi_fmad(float addrspace(1)* %out, float %x, float %y, float %z) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %x.abs = call float @llvm.fabs.f32(float %x)
  %mul2 = fmul fast float %x.abs, 2.0
  %mad0 = fadd fast float %mul2, %y
  %mad1 = fadd fast float %mul2, %z
  store volatile float %mad0, float addrspace(1)* %out
  store volatile float %mad1, float addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}fmul_x2_xn2:
; GCN: v_mul_f32_e64 [[TMP0:v[0-9]+]], [[X:s[0-9]+]], -4.0
; GCN: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[X]], [[TMP0]]
; GCN: buffer_store_dword [[RESULT]]
define void @fmul_x2_xn2(float addrspace(1)* %out, float %x, float %y) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %mul2 = fmul fast float %x, 2.0
  %muln2 = fmul fast float %x, -2.0
  %mul = fmul fast float %mul2, %muln2
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmul_x2_xn3:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0xc0c00000
; GCN: v_mul_f32_e32 [[TMP0:v[0-9]+]], [[X:s[0-9]+]], [[K]]
; GCN: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[X]], [[TMP0]]
; GCN: buffer_store_dword [[RESULT]]
define void @fmul_x2_xn3(float addrspace(1)* %out, float %x, float %y) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %mul2 = fmul fast float %x, 2.0
  %muln2 = fmul fast float %x, -3.0
  %mul = fmul fast float %mul2, %muln2
  store volatile float %mul, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }
attributes #1 = { nounwind readnone }
