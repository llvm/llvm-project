; RUN: not --crash llc < %s -march=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that we fail to select fsin without fast-math enabled

declare float @llvm.sin.f32(float)

; CHECK: LLVM ERROR: Cannot select: {{.*}}: f32 = fsin
; CHECK: In function: test_fsin_safe
define float @test_fsin_safe(float %a) #0 {
  %r = tail call float @llvm.sin.f32(float %a)
  ret float %r
}

attributes #0 = { "unsafe-fp-math" = "false" }
