; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.ddy.fine), %{{.*}} is only supported in shaders.

define noundef float @ddy_fine(float noundef %a) {
entry:
  %spv.ddy.fine = call float @llvm.spv.ddy.fine.f32(float %a)
  ret float %spv.ddy.fine
}

declare float @llvm.spv.ddy.fine.f32(float)
