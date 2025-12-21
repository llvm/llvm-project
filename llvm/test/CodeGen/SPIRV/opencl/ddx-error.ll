; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.ddx), %{{.*}} is only supported in shaders.

define noundef float @ddx(float noundef %a) {
entry:
  %spv.ddx = call float @llvm.spv.ddx.f32(float %a)
  ret float %spv.ddx
}

declare float @llvm.spv.ddx.f32(float)
