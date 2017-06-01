; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare half @llvm.canonicalize.f16(half) #0

; GCN-LABEL: {{^}}v_test_canonicalize_var_f16:
; GCN: v_mul_f16_e32 [[REG:v[0-9]+]], 1.0, {{v[0-9]+}}
; GCN: buffer_store_short [[REG]]
define void @v_test_canonicalize_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_f16:
; GCN: v_mul_f16_e64 [[REG:v[0-9]+]], 1.0, {{s[0-9]+}}
; GCN: buffer_store_short [[REG]]
define void @s_test_canonicalize_var_f16(half addrspace(1)* %out, i16 zeroext %val.arg) #1 {
  %val = bitcast i16 %val.arg to half
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_p0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff8000{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_n0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_p1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffbc00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_n1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4c00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_literal_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 16.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_no_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_no_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff83ff{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7c00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_qnan_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C00)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_qnan_value_neg1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -1 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_qnan_value_neg2_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -2 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_snan0_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_snan1_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7DFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_snan2_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFDFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define void @test_fold_canonicalize_snan3_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFC01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-features"="-flat-for-global" }
attributes #2 = { nounwind "target-features"="-flat-for-global,-fp16-denormals,-fp16-denormals" }
attributes #3 = { nounwind "target-features"="-flat-for-global,+fp16-denormals,+fp64-denormals" }
