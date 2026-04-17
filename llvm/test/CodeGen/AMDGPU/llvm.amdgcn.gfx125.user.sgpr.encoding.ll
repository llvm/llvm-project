; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s -o - | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj -o - | llvm-readobj --notes -

; CHECK:     .sgpr_count:     32
define amdgpu_kernel void @many_inreg_i32(
  i32 inreg %a0, i32 inreg %a1, i32 inreg %a2, i32 inreg %a3,
  i32 inreg %a4, i32 inreg %a5, i32 inreg %a6, i32 inreg %a7,
  i32 inreg %a8, i32 inreg %a9, i32 inreg %a10, i32 inreg %a11,
  i32 inreg %a12, i32 inreg %a13, i32 inreg %a14, i32 inreg %a15,
  i32 inreg %a16, i32 inreg %a17, i32 inreg %a18, i32 inreg %a19,
  i32 inreg %a20, i32 inreg %a21, i32 inreg %a22, i32 inreg %a23,
  i32 inreg %a24, i32 inreg %a25, i32 inreg %a26, i32 inreg %a27,
  i32 inreg %a28, i32 inreg %a29, i32 inreg %a30, i32 inreg %a31,
  i32 inreg %a32, i32 inreg %a33, i32 inreg %a34, i32 inreg %a35) {
  ret void
}
