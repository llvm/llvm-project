; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu10.10 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu10.10 -filetype=null %s 2>&1 | FileCheck %s

; The 33rd inreg argument is in s32, which the callee expects to be the stack
; pointer. GlobalISel used to clobber the argument instead of reporting this.

; CHECK: LLVM ERROR: call in graphics shader with too many input SGPRs

declare amdgpu_gfx void @gfx_func(i32)

define amdgpu_ps void @ps_33_inreg_args(i32 inreg %a0, i32 inreg %a1, i32 inreg %a2, i32 inreg %a3, i32 inreg %a4, i32 inreg %a5, i32 inreg %a6, i32 inreg %a7, i32 inreg %a8, i32 inreg %a9, i32 inreg %a10, i32 inreg %a11, i32 inreg %a12, i32 inreg %a13, i32 inreg %a14, i32 inreg %a15, i32 inreg %a16, i32 inreg %a17, i32 inreg %a18, i32 inreg %a19, i32 inreg %a20, i32 inreg %a21, i32 inreg %a22, i32 inreg %a23, i32 inreg %a24, i32 inreg %a25, i32 inreg %a26, i32 inreg %a27, i32 inreg %a28, i32 inreg %a29, i32 inreg %a30, i32 inreg %a31, i32 inreg %a32) {
  call amdgpu_gfx void @gfx_func(i32 %a32)
  ret void
}
