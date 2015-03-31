; RUN: llc < %s -march=amdgcn -mcpu=SI | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga | FileCheck %s

define void @test_load_store(half addrspace(1)* %in, half addrspace(1)* %out) {
; CHECK-LABEL: {{^}}test_load_store:
; CHECK: buffer_load_ushort [[TMP:v[0-9]+]]
; CHECK: buffer_store_short [[TMP]]
  %val = load half addrspace(1)* %in
  store half %val, half addrspace(1) * %out
  ret void
}

define void @test_bitcast_from_half(half addrspace(1)* %in, i16 addrspace(1)* %out) {
; CHECK-LABEL: {{^}}test_bitcast_from_half:
; CHECK: buffer_load_ushort [[TMP:v[0-9]+]]
; CHECK: buffer_store_short [[TMP]]
  %val = load half addrspace(1) * %in
  %val_int = bitcast half %val to i16
  store i16 %val_int, i16 addrspace(1)* %out
  ret void
}

define void @test_bitcast_to_half(half addrspace(1)* %out, i16 addrspace(1)* %in) {
; CHECK-LABEL: {{^}}test_bitcast_to_half:
; CHECK: buffer_load_ushort [[TMP:v[0-9]+]]
; CHECK: buffer_store_short [[TMP]]
  %val = load i16 addrspace(1)* %in
  %val_fp = bitcast i16 %val to half
  store half %val_fp, half addrspace(1)* %out
  ret void
}

define void @test_extend32(half addrspace(1)* %in, float addrspace(1)* %out) {
; CHECK-LABEL: {{^}}test_extend32:
; CHECK: v_cvt_f32_f16_e32

  %val16 = load half addrspace(1)* %in
  %val32 = fpext half %val16 to float
  store float %val32, float addrspace(1)* %out
  ret void
}

define void @test_extend64(half addrspace(1)* %in, double addrspace(1)* %out) {
; CHECK-LABEL: {{^}}test_extend64:
; CHECK: v_cvt_f32_f16_e32
; CHECK: v_cvt_f64_f32_e32

  %val16 = load half addrspace(1)* %in
  %val64 = fpext half %val16 to double
  store double %val64, double addrspace(1)* %out
  ret void
}

define void @test_trunc32(float addrspace(1)* %in, half addrspace(1)* %out) {
; CHECK-LABEL: {{^}}test_trunc32:
; CHECK: v_cvt_f16_f32_e32

  %val32 = load float addrspace(1)* %in
  %val16 = fptrunc float %val32 to half
  store half %val16, half addrspace(1)* %out
  ret void
}
