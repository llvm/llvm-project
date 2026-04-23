; RUN: not llc -mtriple=nvptx64 -mcpu=sm_80 -filetype=null %s 2>&1 | FileCheck %s

; Verify that an fp.control bundle with an unsupported rounding mode ("rmm",
; round-to-nearest-ties-away) produces a clean diagnostic error rather than
; hitting an llvm_unreachable in the backend.

; CHECK: unsupported rounding mode 'rmm' in fp.control bundle for target 'nvptx64'

define float @test_fadd_rmm(float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b)
      [ "fp.control"(metadata !"rmm") ]
  ret float %r
}

declare float @llvm.fadd.f32(float, float)
