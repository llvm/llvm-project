; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; CHECK-LABEL test_fabsf(
define float @test_fabsf(float %f) {
; CHECK: abs.f32
  %x = call float @llvm.fabs.f32(float %f)
  ret float %x
}

; CHECK-LABEL: test_fabs(
define double @test_fabs(double %d) {
; CHECK: abs.f64
  %x = call double @llvm.fabs.f64(double %d)
  ret double %x
}

; CHECK-LABEL: test_nvvm_sqrt(
define float @test_nvvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.nvvm.sqrt.f(float %a)
  ret float %val
}

; CHECK-LABEL: test_llvm_sqrt(
define float @test_llvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.sqrt.f32(float %a)
  ret float %val
}

; CHECK-LABEL: test_bitreverse32(
define i32 @test_bitreverse32(i32 %a) {
; CHECK: brev.b32
  %val = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %val
}

; CHECK-LABEL: test_bitreverse64(
define i64 @test_bitreverse64(i64 %a) {
; CHECK: brev.b64
  %val = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %val
}

; CHECK-LABEL: test_popc32(
define i32 @test_popc32(i32 %a) {
; CHECK: popc.b32
  %val = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %val
}

; CHECK-LABEL: test_popc64
define i64 @test_popc64(i64 %a) {
; CHECK: popc.b64
; CHECK: cvt.u64.u32
  %val = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %val
}

; NVPTX popc.b64 returns an i32 even though @llvm.ctpop.i64 returns an i64, so
; if this function returns an i32, there's no need to do any type conversions
; in the ptx.
; CHECK-LABEL: test_popc64_trunc
define i32 @test_popc64_trunc(i64 %a) {
; CHECK: popc.b64
; CHECK-NOT: cvt.
  %val = call i64 @llvm.ctpop.i64(i64 %a)
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}

; llvm.ctpop.i16 is implemenented by converting to i32, running popc.b32, and
; then converting back to i16.
; CHECK-LABEL: test_popc16
define void @test_popc16(i16 %a, i16* %b) {
; CHECK: cvt.u32.u16
; CHECK: popc.b32
; CHECK: cvt.u16.u32
  %val = call i16 @llvm.ctpop.i16(i16 %a)
  store i16 %val, i16* %b
  ret void
}

; If we call llvm.ctpop.i16 and then zext the result to i32, we shouldn't need
; to do any conversions after calling popc.b32, because that returns an i32.
; CHECK-LABEL: test_popc16_to_32
define i32 @test_popc16_to_32(i16 %a) {
; CHECK: cvt.u32.u16
; CHECK: popc.b32
; CHECK-NOT: cvt.
  %val = call i16 @llvm.ctpop.i16(i16 %a)
  %zext = zext i16 %val to i32
  ret i32 %zext
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare float @llvm.nvvm.sqrt.f(float)
declare float @llvm.sqrt.f32(float)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
