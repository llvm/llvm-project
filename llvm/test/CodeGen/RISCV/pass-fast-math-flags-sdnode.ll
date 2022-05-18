; REQUIRES: asserts
; RUN: llc < %s -mtriple=riscv64 -mattr=+v -debug-only=isel -o /dev/null 2>&1                        | FileCheck %s
declare <vscale x 1 x double> @llvm.vp.fmul.nxv1f64(<vscale x 1 x double> %x, <vscale x 1 x double> %y, <vscale x 1 x i1> %m, i32 %vl)

define <vscale x 1 x double> @foo(<vscale x 1 x double> %x, <vscale x 1 x double> %y, <vscale x 1 x double> %z, <vscale x 1 x i1> %m, i32 %vl) {
; CHECK:      t14: nxv1f64 = vp_fmul nnan ninf nsz arcp contract afn reassoc t2, t4, t8, t13
  %1 = call fast <vscale x 1 x double> @llvm.vp.fmul.nxv1f64(<vscale x 1 x double> %x, <vscale x 1 x double> %y, <vscale x 1 x i1> %m, i32 %vl)
  ret <vscale x 1 x double> %1
}
