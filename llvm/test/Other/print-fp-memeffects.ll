; RUN: opt %s -print-fp-memory-effects -S | FileCheck %s --match-full-lines

define float @test_00(float %x) {
  %res = call float @llvm.nearbyint.f32(float %x) [ "fpe.round"(metadata !"upward"), "fpe.except"(metadata !"ignore") ]
  ret float %res
; CHECK-LABEL: define float @test_00({{.*}}
; CHECK:       {{.*}} = call float @llvm.nearbyint.f32({{.*}}]
}

define float @test_01(float %x) strictfp {
  %res = call float @llvm.nearbyint.f32(float %x) [ "fpe.round"(metadata !"upward"), "fpe.except"(metadata !"strict") ]
  ret float %res
; CHECK-LABEL: define float @test_01({{.*}}
; CHECK:       {{.*}} = call float @llvm.nearbyint.f32({{.*}}] ; fpe=[rw]
}
