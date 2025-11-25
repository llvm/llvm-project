; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.ddy.coarse), %{{.*}} is only supported in shaders.

define noundef float @ddy_coarse(float noundef %a) {
entry:
  %spv.ddy.coarse = call float @llvm.spv.ddy.coarse.f32(float %a)
  ret float %spv.ddy.coarse
}

declare float @llvm.spv.ddy.coarse.f32(float)
