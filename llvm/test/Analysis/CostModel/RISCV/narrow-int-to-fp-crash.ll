; Regression test for https://github.com/llvm/llvm-project/issues/236321

; RUN: opt -mtriple=riscv64 -mattr=+v,+zfh,+zvfh -passes='print<cost-model>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=RV64ZVFH
; RUN: opt -mtriple=riscv64 -mattr=+v,+zfh,+zvfhmin -passes='print<cost-model>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=RV64ZVFHMIN
; RUN: opt -mtriple=riscv64 -mattr=+v,+zfh,+zvfh,+zfbfmin,+zvfbfmin -passes='print<cost-model>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=RV64BF

define <3 x half> @sitofp_v3i2_to_v3half(<3 x i2> %x) {
; RV64ZVFH-LABEL: 'sitofp_v3i2_to_v3half'
; RV64ZVFH: sitofp <3 x i2> %x to <3 x half>
; RV64ZVFHMIN-LABEL: 'sitofp_v3i2_to_v3half'
; RV64ZVFHMIN: sitofp <3 x i2> %x to <3 x half>
; RV64BF-LABEL: 'sitofp_v3i2_to_v3half'
; RV64BF: sitofp <3 x i2> %x to <3 x half>
  %r = sitofp <3 x i2> %x to <3 x half>
  ret <3 x half> %r
}

define <3 x half> @uitofp_v3i4_to_v3half(<3 x i4> %x) {
; RV64ZVFH-LABEL: 'uitofp_v3i4_to_v3half'
; RV64ZVFH: uitofp <3 x i4> %x to <3 x half>
; RV64ZVFHMIN-LABEL: 'uitofp_v3i4_to_v3half'
; RV64ZVFHMIN: uitofp <3 x i4> %x to <3 x half>
; RV64BF-LABEL: 'uitofp_v3i4_to_v3half'
; RV64BF: uitofp <3 x i4> %x to <3 x half>
  %r = uitofp <3 x i4> %x to <3 x half>
  ret <3 x half> %r
}

define <vscale x 2 x half> @sitofp_nxv2i2_to_nxv2half(<vscale x 2 x i2> %x) {
; RV64ZVFH-LABEL: 'sitofp_nxv2i2_to_nxv2half'
; RV64ZVFH: sitofp <vscale x 2 x i2> %x to <vscale x 2 x half>
; RV64ZVFHMIN-LABEL: 'sitofp_nxv2i2_to_nxv2half'
; RV64ZVFHMIN: sitofp <vscale x 2 x i2> %x to <vscale x 2 x half>
; RV64BF-LABEL: 'sitofp_nxv2i2_to_nxv2half'
; RV64BF: sitofp <vscale x 2 x i2> %x to <vscale x 2 x half>
  %r = sitofp <vscale x 2 x i2> %x to <vscale x 2 x half>
  ret <vscale x 2 x half> %r
}

define <3 x bfloat> @sitofp_v3i2_to_v3bf16(<3 x i2> %x) {
; RV64BF-LABEL: 'sitofp_v3i2_to_v3bf16'
; RV64BF: sitofp <3 x i2> %x to <3 x bfloat>
  %r = sitofp <3 x i2> %x to <3 x bfloat>
  ret <3 x bfloat> %r
}
