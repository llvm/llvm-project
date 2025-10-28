; RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 < %s | FileCheck %s

define amdgpu_ps i16 @abs_i16(i16 inreg %arg) {
; CHECK-LABEL: abs_i16:
; CHECK: %bb.0:
; CHECK-NEXT: s_sext_i32_i16 s0, s0
; CHECK-NEXT: s_abs_i32 s0, s0

  %res = call i16 @llvm.abs.i16(i16 %arg, i1 false)
  ret i16 %res
}

define amdgpu_ps i16 @abs_i16_neg(i16 inreg %arg) {
; CHECK-LABEL: abs_i16_neg:
; CHECK: ; %bb.0:
; CHECK-NEXT: s_sext_i32_i16 s0, s0
; CHECK-NEXT: s_abs_i32 s0, s0
; CHECK-NEXT: s_sub_i32 s0, 0, s0
  %res1 = call i16 @llvm.abs.i16(i16 %arg, i1 false)
  %res2 = sub i16 0, %res1
  ret i16 %res2
}