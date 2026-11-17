; RUN: not llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_20 -o /dev/null 2>&1 | FileCheck %s

; sm_2x has no bindless textures or surfaces, so an image handle that is not a
; kernel parameter or a global must be diagnosed.

; CHECK: error: {{.*}}in function tex_from_call {{.*}}: image handle must be a kernel parameter or a global on sm_2x
; CHECK: error: {{.*}}in function surf_from_call {{.*}}: image handle must be a kernel parameter or a global on sm_2x

declare { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64, i32)
declare i32 @llvm.nvvm.suld.1d.i32.trap(i64, i32)
declare i64 @get_handle()

define ptx_kernel void @tex_from_call(ptr %out, i32 %idx) {
  %h = call i64 @get_handle()
  %v = call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %h, i32 %idx)
  %r = extractvalue { float, float, float, float } %v, 0
  store float %r, ptr %out
  ret void
}

define ptx_kernel void @surf_from_call(ptr %out, i32 %idx) {
  %h = call i64 @get_handle()
  %v = call i32 @llvm.nvvm.suld.1d.i32.trap(i64 %h, i32 %idx)
  store i32 %v, ptr %out
  ret void
}
