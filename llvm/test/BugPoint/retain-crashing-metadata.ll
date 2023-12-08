; REQUIRES: plugins
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t-notype -bugpoint-crashmetadata -silence-passes > /dev/null
; RUN: llvm-dis %t-notype-reduced-simplified.bc -o - | FileCheck %s
;
; Make sure BugPoint retains metadata contributing to a crash.

; CHECK-LABEL: define void @test2(float %f) {
; CHECK-NEXT: %arg = fadd float %f, 1.000000e+01
; CHECK-NOT: !fpmath
; CHECK-NEXT: %x = call float @llvm.fabs.f32(float %arg), !fpmath [[FPMATH:![0-9]+]]
; CHECK-NEXT: ret void

; CHECK: [[FPMATH]] = !{float 2.500000e+00}
define void @test2(float %f) {
    %arg = fadd float %f, 1.000000e+01, !fpmath !0
    %x = call float @llvm.fabs.f32(float %arg), !fpmath !0
    ret void
}

declare float @llvm.fabs.f32(float)

!0 = !{float 2.500000e+00}
