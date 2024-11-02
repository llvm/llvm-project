; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] rint

define dso_local spir_func float @foo(float %x) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.nearbyint.f32(float %x)
  ret float %0
}

declare float @llvm.nearbyint.f32(float)
