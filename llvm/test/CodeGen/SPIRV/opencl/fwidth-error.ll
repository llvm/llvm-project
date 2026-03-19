; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.fwidth), %{{.*}} is only supported in shaders.

define noundef float @fwidth(float noundef %a) {
entry:
  %spv.fwidth = call float @llvm.spv.fwidth.f32(float %a)
  ret float %spv.fwidth
}

declare float @llvm.spv.fwidth.f32(float)
