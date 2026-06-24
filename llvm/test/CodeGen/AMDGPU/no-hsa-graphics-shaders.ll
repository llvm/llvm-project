; RUN: not llc -mtriple=amdgcn-unknown-amdhsa -O0 -filetype=null < %s 2>&1 | FileCheck %s

@I = global i32 42
@P = global ptr @I

; CHECK: error: <unknown>:0:0: in function pixel_shader_zero_args void (): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function pixel_shader_one_arg void (ptr): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function pixel_shader_two_args void (ptr, i32): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function vertex_shader_zero_args void (): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function vertex_shader_one_arg void (ptr): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function vertex_shader_two_args void (ptr, i32): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function geometry_shader_zero_args void (): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function geometry_shader_one_arg void (ptr): unsupported non-compute shaders with HSA
; CHECK: error: <unknown>:0:0: in function geometry_shader_two_args void (ptr, i32): unsupported non-compute shaders with HSA

define amdgpu_ps void @pixel_shader_zero_args() {
  %i = load i32, ptr @I
  store i32 %i, ptr @P
  ret void
}

define amdgpu_ps void @pixel_shader_one_arg(ptr %p) {
  %i = load i32, ptr @I
  store i32 %i, ptr %p
  ret void
}

define amdgpu_ps void @pixel_shader_two_args(ptr %p, i32 %i) {
  store i32 %i, ptr %p
  ret void
}

define amdgpu_vs void @vertex_shader_zero_args() {
  %i = load i32, ptr @I
  store i32 %i, ptr @P
  ret void
}

define amdgpu_vs void @vertex_shader_one_arg(ptr %p) {
  %i = load i32, ptr @I
  store i32 %i, ptr %p
  ret void
}

define amdgpu_vs void @vertex_shader_two_args(ptr %p, i32 %i) {
  store i32 %i, ptr %p
  ret void
}

define amdgpu_gs void @geometry_shader_zero_args() {
  %i = load i32, ptr @I
  store i32 %i, ptr @P
  ret void
}

define amdgpu_gs void @geometry_shader_one_arg(ptr %p) {
  %i = load i32, ptr @I
  store i32 %i, ptr %p
  ret void
}

define amdgpu_gs void @geometry_shader_two_args(ptr %p, i32 %i) {
  store i32 %i, ptr %p
  ret void
}
