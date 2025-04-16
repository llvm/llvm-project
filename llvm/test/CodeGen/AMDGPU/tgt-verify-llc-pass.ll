; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -enable-new-pm %s -o - 2>&1 | FileCheck %s --allow-empty

define amdgpu_cs void @void_shader() {
; CHECK-NOT: Shaders must return void
  ret void
}
