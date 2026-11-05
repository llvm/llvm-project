; RUN: not --crash llc -mtriple=nvptx64 -filetype=null %s 2>&1 | FileCheck %s

declare float @llvm.exp.f32(float)

; CHECK: error: no libcall available for fexp
define float @exp_no_afn(float %x) {
  %result = call float @llvm.exp.f32(float %x)
  ret float %result
}
