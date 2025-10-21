; RUN: not --crash llc < %s -mtriple=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that we fail to select fsin without fast-math enabled

declare float @llvm.sin.f32(float)

; CHECK: LLVM ERROR: Cannot select: {{.*}}: f32 = fsin
; CHECK: In function: test_fsin_safe
define float @test_fsin_safe(float %a) {
  %r = tail call float @llvm.sin.f32(float %a)
  ret float %r
}
