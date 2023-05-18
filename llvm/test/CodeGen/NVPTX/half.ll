; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

; CHECK: .b8 half_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
@"half_array" = addrspace(1) constant [4 x half]
                [half 0xH0201, half 0xH0403, half 0xH0605, half 0xH0807]

define void @test_load_store(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_load_store
; CHECK: ld.global.b16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.b16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load half, ptr addrspace(1) %in
  store half %val, ptr addrspace(1) %out
  ret void
}

define void @test_bitcast_from_half(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_bitcast_from_half
; CHECK: ld.global.b16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.b16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load half, ptr addrspace(1) %in
  %val_int = bitcast half %val to i16
  store i16 %val_int, ptr addrspace(1) %out
  ret void
}

define void @test_bitcast_to_half(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; CHECK-LABEL: @test_bitcast_to_half
; CHECK: ld.global.u16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.u16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load i16, ptr addrspace(1) %in
  %val_fp = bitcast i16 %val to half
  store half %val_fp, ptr addrspace(1) %out
  ret void
}

define void @test_extend32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_extend32
; CHECK: cvt.f32.f16

  %val16 = load half, ptr addrspace(1) %in
  %val32 = fpext half %val16 to float
  store float %val32, ptr addrspace(1) %out
  ret void
}

define void @test_extend64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_extend64
; CHECK: cvt.f64.f16

  %val16 = load half, ptr addrspace(1) %in
  %val64 = fpext half %val16 to double
  store double %val64, ptr addrspace(1) %out
  ret void
}

define void @test_trunc32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: test_trunc32
; CHECK: cvt.rn.f16.f32

  %val32 = load float, ptr addrspace(1) %in
  %val16 = fptrunc float %val32 to half
  store half %val16, ptr addrspace(1) %out
  ret void
}

define void @test_trunc64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_trunc64
; CHECK: cvt.rn.f16.f64

  %val32 = load double, ptr addrspace(1) %in
  %val16 = fptrunc double %val32 to half
  store half %val16, ptr addrspace(1) %out
  ret void
}
