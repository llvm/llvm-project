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

declare i32 @llvm.pisa.lane.id()
declare i32 @llvm.pisa.subgroup.size()
declare i32 @llvm.pisa.work.dim()
declare float @llvm.pisa.cas.fatom.f32.p0(ptr, float, float, i8 immarg)
