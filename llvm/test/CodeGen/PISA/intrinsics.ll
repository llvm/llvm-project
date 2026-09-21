; Verify that the PISA target intrinsics are registered in the IR layer and
; round-trip through the verifier/assembler.

; RUN: opt -S < %s | FileCheck %s

define void @test() {
; CHECK-LABEL: define void @test()
  %lane = call i32 @llvm.pisa.lane.id()
; CHECK: call i32 @llvm.pisa.lane.id()
  %sgsize = call i32 @llvm.pisa.subgroup.size()
; CHECK: call i32 @llvm.pisa.subgroup.size()
  %wdim = call i32 @llvm.pisa.work.dim()
; CHECK: call i32 @llvm.pisa.work.dim()
  ret void
}

define float @test_cas_fatom(ptr %addr, float %compare, float %value) {
; CHECK-LABEL: define float @test_cas_fatom(
  %result = call float @llvm.pisa.cas.fatom.f32.p0(ptr %addr, float %compare, float %value, i8 2)
; CHECK: call float @llvm.pisa.cas.fatom.f32.p0(ptr %addr, float %compare, float %value, /* order=monotonic */ i8 2)
  ret float %result
}

define i32 @test_rounding_mode_printer(float %x, float %y) {
; CHECK-LABEL: define i32 @test_rounding_mode_printer(
  %sqrt = call float @llvm.pisa.fsqrt.rnd.f32(float %x, i8 0)
; CHECK: call float @llvm.pisa.fsqrt.rnd.f32(float %x, /* round=.rz */ i8 0)
  %div = call float @llvm.pisa.fdiv.rnd.f32(float %x, float %y, i8 1)
; CHECK: call float @llvm.pisa.fdiv.rnd.f32(float %x, float %y, /* round=.re */ i8 1)
  %conv = call i32 @llvm.pisa.fptosi.rnd.i32.f32(float %div, i8 2)
; CHECK: call i32 @llvm.pisa.fptosi.rnd.i32.f32(float %div, /* round=.ru */ i8 2)
  ret i32 %conv
}

declare i32 @llvm.pisa.lane.id()
declare i32 @llvm.pisa.subgroup.size()
declare i32 @llvm.pisa.work.dim()
declare float @llvm.pisa.cas.fatom.f32.p0(ptr, float, float, i8 immarg)
declare float @llvm.pisa.fsqrt.rnd.f32(float, i8 immarg)
declare float @llvm.pisa.fdiv.rnd.f32(float, float, i8 immarg)
declare i32 @llvm.pisa.fptosi.rnd.i32.f32(float, i8 immarg)
