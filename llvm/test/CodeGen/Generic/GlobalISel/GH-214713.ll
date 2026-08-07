; REQUIRES: asserts, aarch64-registered-target
; RUN: not --crash llc -global-isel -stop-after=irtranslator <%s 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK: Assertion {{.*}}ResTy.isVector() && "Expected vector result type"

define float @test(<2 x float> %w) {
  %1 = call { <1 x float>, <1 x float> } @llvm.vector.deinterleave2.v2f32(<2 x float> %w)
  %2 = extractvalue { <1 x float>, <1 x float> } %1, 0
  %3 = extractvalue { <1 x float>, <1 x float> } %1, 1
  %4 = extractelement <1 x float> %2, i32 0
  %5 = extractelement <1 x float> %3, i32 0
  %6 = fadd float %4, %5
  ret float %6
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { <1 x float>, <1 x float> } @llvm.vector.deinterleave2.v2f32(<2 x float>) #1
