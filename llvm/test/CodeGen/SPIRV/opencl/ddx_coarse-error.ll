; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.ddx.coarse), %{{.*}} is only supported in shaders.

define noundef float @ddx_coarse(float noundef %a) {
entry:
  %spv.ddx.coarse = call float @llvm.spv.ddx.coarse.f32(float %a)
  ret float %spv.ddx.coarse
}

declare float @llvm.spv.ddx.coarse.f32(float)
