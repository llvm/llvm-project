; RUN: not llc < %s -mcpu=sm_30 -o /dev/null 2>&1 | FileCheck %s

; In independent texture mode (NVCL), PTX does not allow texture or sampler
; handles in registers, so a handle that is not a kernel parameter or a global
; must be diagnosed.

; CHECK: error: {{.*}}in function tex_from_call {{.*}}: texture or sampler handle must be a kernel parameter or a global in independent texture mode
; CHECK: error: {{.*}}in function sampler_from_call {{.*}}: texture or sampler handle must be a kernel parameter or a global in independent texture mode

target triple = "nvptx64-unknown-nvcl"

declare { float, float, float, float } @llvm.nvvm.tex.1d.v4f32.s32(i64, i64, i32)
declare i64 @get_handle()

define ptx_kernel void @tex_from_call(i64 %sampler, ptr %out, i32 %idx) {
  %h = call i64 @get_handle()
  %v = call { float, float, float, float } @llvm.nvvm.tex.1d.v4f32.s32(i64 %h, i64 %sampler, i32 %idx)
  %r = extractvalue { float, float, float, float } %v, 0
  store float %r, ptr %out
  ret void
}

define ptx_kernel void @sampler_from_call(i64 %img, ptr %out, i32 %idx) {
  %s = call i64 @get_handle()
  %v = call { float, float, float, float } @llvm.nvvm.tex.1d.v4f32.s32(i64 %img, i64 %s, i32 %idx)
  %r = extractvalue { float, float, float, float } %v, 0
  store float %r, ptr %out
  ret void
}

!nvvm.annotations = !{!2, !3}
!2 = !{ptr @sampler_from_call, !"rdoimage", i32 0}
!3 = !{ptr @tex_from_call, !"sampler", i32 0}
