; RUN: llc < %s -mtriple=thumbv7-apple-ios7.0 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-OLD
; RUN: llc < %s -mtriple=thumbv7s-apple-ios7.0 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-F16
; RUN: llc < %s -mtriple=thumbv8-apple-ios7.0 | FileCheck %s --check-prefix=CHECK  --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=armv8r-none-none-eabi | FileCheck %s --check-prefix=CHECK  --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=armv8r-none-none-eabi -mattr=-fp64 | FileCheck %s --check-prefix=CHECK  --check-prefix=CHECK-V8-SP
; RUN: llc < %s -mtriple=armv8.1m-none-none-eabi -mattr=+fp-armv8 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=armv8.1m-none-none-eabi -mattr=+fp-armv8,-fp64 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8-SP
; RUN: llc < %s -mtriple=armv8.1m-none-none-eabi -mattr=+mve.fp,+fp64 | FileCheck %s --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=armv8.1m-none-none-eabi -mattr=+mve.fp | FileCheck %s --check-prefix=CHECK-V8-SP

define void @test_load_store(ptr %in, ptr %out) {
; CHECK-LABEL: test_load_store:
; CHECK: ldrh [[TMP:r[0-9]+]], [r0]
; CHECK: strh [[TMP]], [r1]
  %val = load half, ptr %in
  store half %val, ptr %out
  ret void
}

define i16 @test_bitcast_from_half(ptr %addr) {
; CHECK-LABEL: test_bitcast_from_half:
; CHECK: ldrh r0, [r0]
  %val = load half, ptr %addr
  %val_int = bitcast half %val to i16
  ret i16 %val_int
}

define void @test_bitcast_to_half(ptr %addr, i16 %in) {
; CHECK-LABEL: test_bitcast_to_half:
; CHECK: strh r1, [r0]
  %val_fp = bitcast i16 %in to half
  store half %val_fp, ptr %addr
  ret void
}

define float @test_extend32(ptr %addr) {
; CHECK-LABEL: test_extend32:

; CHECK-OLD: b.w ___extendhfsf2
; CHECK-F16: vcvtb.f32.f16
; CHECK-V8: vcvtb.f32.f16
; CHECK-V8-SP: vcvtb.f32.f16
  %val16 = load half, ptr %addr
  %val32 = fpext half %val16 to float
  ret float %val32
}

define double @test_extend64(ptr %addr) {
; CHECK-LABEL: test_extend64:

; CHECK-OLD: bl ___extendhfsf2
; CHECK-OLD: vcvt.f64.f32
; CHECK-F16: vcvtb.f32.f16
; CHECK-F16: vcvt.f64.f32
; CHECK-V8: vcvtb.f64.f16
; CHECK-V8-SP: vcvtb.f32.f16
; CHECK-V8-SP: bl __aeabi_f2d
  %val16 = load half, ptr %addr
  %val32 = fpext half %val16 to double
  ret double %val32
}

define void @test_trunc32(float %in, ptr %addr) {
; CHECK-LABEL: test_trunc32:

; CHECK-OLD: bl ___truncsfhf2
; CHECK-F16: vcvtb.f16.f32
; CHECK-V8: vcvtb.f16.f32
; CHECK-V8-SP: vcvtb.f16.f32
  %val16 = fptrunc float %in to half
  store half %val16, ptr %addr
  ret void
}

define void @test_trunc64(double %in, ptr %addr) {
; CHECK-LABEL: test_trunc64:

; CHECK-OLD: bl ___truncdfhf2
; CHECK-F16: bl ___truncdfhf2
; CHECK-V8: vcvtb.f16.f64
; CHECK-V8-SP: bl __aeabi_d2h
  %val16 = fptrunc double %in to half
  store half %val16, ptr %addr
  ret void
}
