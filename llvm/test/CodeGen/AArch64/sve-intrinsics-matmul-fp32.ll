; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+sve,+f32mm -asm-verbose=0 < %s -o - | FileCheck %s

define <vscale x 4 x float> @fmmla_s(<vscale x 4 x float> %r, <vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
entry:
; CHECK-LABEL: fmmla_s:
; CHECK-NEXT:  fmmla   z0.s, z1.s, z2.s
; CHECK-NEXT:  ret
  %val = tail call <vscale x 4 x float> @llvm.aarch64.sve.fmmla.nxv4f32(<vscale x 4 x float> %r, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %val
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fmmla.nxv4f32(<vscale x 4 x float>,<vscale x 4 x float>,<vscale x 4 x float>)

