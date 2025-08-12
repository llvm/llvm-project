; RUN: not --crash llc < %s -mtriple=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that we fail to select fcos without fast-math enabled

declare float @llvm.cos.f32(float)

; CHECK: LLVM ERROR: Cannot select: {{.*}}: f32 = fcos
; CHECK: In function: test_fcos_safe
define float @test_fcos_safe(float %a) #0 {
  %r = tail call float @llvm.cos.f32(float %a)
  ret float %r
}

attributes #0 = { "unsafe-fp-math" = "false" }
