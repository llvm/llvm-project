; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-verify-tgt %s -o - 2>&1 | FileCheck %s --allow-empty

define amdgpu_cs void @void_shader() {
; CHECK-NOT: Shaders must return void
  ret void
}
