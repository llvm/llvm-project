; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -o - %s 2>&1 | FileCheck -check-prefix=SDAG-ERR %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -o - %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; SDAG-ERR: LLVM ERROR: unsupported libcall legalization
; GISEL-ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(s128) = G_SDIV %{{[0-9]+}}:_, %{{[0-9]+}}:_ (in function: v_sdiv_v2i128_vv)

define <2 x i128> @v_sdiv_v2i128_vv(<2 x i128> %lhs, <2 x i128> %rhs) {
  %shl = sdiv <2 x i128> %lhs, %rhs
  ret <2 x i128> %shl
}

define <2 x i128> @v_udiv_v2i128_vv(<2 x i128> %lhs, <2 x i128> %rhs) {
  %shl = udiv <2 x i128> %lhs, %rhs
  ret <2 x i128> %shl
}

define <2 x i128> @v_srem_v2i128_vv(<2 x i128> %lhs, <2 x i128> %rhs) {
  %shl = srem <2 x i128> %lhs, %rhs
  ret <2 x i128> %shl
}

define <2 x i128> @v_urem_v2i128_vv(<2 x i128> %lhs, <2 x i128> %rhs) {
  %shl = urem <2 x i128> %lhs, %rhs
  ret <2 x i128> %shl
}
