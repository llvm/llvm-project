; RUN: not --crash llc < %s -mtriple=nvptx64 2>&1 | FileCheck %s

declare float @llvm.powi.f32.i32(float, i32)

; CHECK: LLVM ERROR: Cannot select: {{.*}}: f32 = fpow
; CHECK: In function: test_powi
define float @test_powi(float %a, i32 %b) {
  %r = call float @llvm.powi.f32.i32(float %a, i32 %b)
  ret float %r
}
