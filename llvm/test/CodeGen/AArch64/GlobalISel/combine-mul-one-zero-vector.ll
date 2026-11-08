; RUN: llc -mtriple=aarch64-linux-gnu -O2 -global-isel -global-isel-abort=1 %s -o - | FileCheck %s

define <2 x i32> @mul_mask_rhs(<2 x i32> %a) {
; CHECK-LABEL: mul_mask_rhs:
; CHECK-NOT: adrp
; CHECK: mov v0.s[1], wzr
; CHECK: ret
  %r = mul <2 x i32> %a, <i32 1, i32 0>
  ret <2 x i32> %r
}

define <2 x i32> @mul_mask_lhs(<2 x i32> %a) {
; CHECK-LABEL: mul_mask_lhs:
; CHECK-NOT: adrp
; CHECK: mov v0.s[1], wzr
; CHECK: ret
  %r = mul <2 x i32> <i32 1, i32 0>, %a
  ret <2 x i32> %r
}

define <2 x i32> @sdiv_const_vec(<2 x i32> %a) {
; CHECK-LABEL: sdiv_const_vec:
; CHECK: mov v0.s[1], wzr
; CHECK-NOT: mla
; CHECK: ret
  %r = sdiv <2 x i32> %a, <i32 7, i32 6>
  ret <2 x i32> %r
}
