; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-verify-tgt -o - < %s 2>&1 | FileCheck %s

define amdgpu_cs i32 @nonvoid_shader() {
; CHECK: Shaders must return void
  ret i32 0
}
