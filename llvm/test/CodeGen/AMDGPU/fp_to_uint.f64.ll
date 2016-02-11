; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; SI-LABEL: {{^}}fp_to_uint_i32_f64:
; SI: v_cvt_u32_f64_e32
define void @fp_to_uint_i32_f64(i32 addrspace(1)* %out, double %in) {
  %cast = fptoui double %in to i32
  store i32 %cast, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @fp_to_uint_v2i32_v2f64
; SI: v_cvt_u32_f64_e32
; SI: v_cvt_u32_f64_e32
define void @fp_to_uint_v2i32_v2f64(<2 x i32> addrspace(1)* %out, <2 x double> %in) {
  %cast = fptoui <2 x double> %in to <2 x i32>
  store <2 x i32> %cast, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @fp_to_uint_v4i32_v4f64
; SI: v_cvt_u32_f64_e32
; SI: v_cvt_u32_f64_e32
; SI: v_cvt_u32_f64_e32
; SI: v_cvt_u32_f64_e32
define void @fp_to_uint_v4i32_v4f64(<4 x i32> addrspace(1)* %out, <4 x double> %in) {
  %cast = fptoui <4 x double> %in to <4 x i32>
  store <4 x i32> %cast, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @fp_to_uint_i64_f64
; CI-DAG: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; CI-DAG: v_trunc_f64_e32 [[TRUNC:v\[[0-9]+:[0-9]+\]]], [[VAL]]
; CI-DAG: s_mov_b32 s[[K0_LO:[0-9]+]], 0{{$}}
; CI-DAG: s_mov_b32 s[[K0_HI:[0-9]+]], 0x3df00000

; CI-DAG: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], [[VAL]], s{{\[}}[[K0_LO]]:[[K0_HI]]{{\]}}
; CI-DAG: v_floor_f64_e32 [[FLOOR:v\[[0-9]+:[0-9]+\]]], [[MUL]]

; CI-DAG: s_mov_b32 s[[K1_HI:[0-9]+]], 0xc1f00000

; CI-DAG: v_fma_f64 [[FMA:v\[[0-9]+:[0-9]+\]]], [[FLOOR]], s{{\[[0-9]+}}:[[K1_HI]]{{\]}}, [[TRUNC]]
; CI-DAG: v_cvt_u32_f64_e32 v[[LO:[0-9]+]], [[FMA]]
; CI-DAG: v_cvt_u32_f64_e32 v[[HI:[0-9]+]], [[FLOOR]]
; CI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @fp_to_uint_i64_f64(i64 addrspace(1)* %out, double addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid
  %val = load double, double addrspace(1)* %gep, align 8
  %cast = fptoui double %val to i64
  store i64 %cast, i64 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @fp_to_uint_v2i64_v2f64
define void @fp_to_uint_v2i64_v2f64(<2 x i64> addrspace(1)* %out, <2 x double> %in) {
  %cast = fptoui <2 x double> %in to <2 x i64>
  store <2 x i64> %cast, <2 x i64> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @fp_to_uint_v4i64_v4f64
define void @fp_to_uint_v4i64_v4f64(<4 x i64> addrspace(1)* %out, <4 x double> %in) {
  %cast = fptoui <4 x double> %in to <4 x i64>
  store <4 x i64> %cast, <4 x i64> addrspace(1)* %out, align 32
  ret void
}
