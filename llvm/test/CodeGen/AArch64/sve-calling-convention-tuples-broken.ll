; RUN: not --crash llc < %s -mtriple aarch64-linux-gnu -mattr=+sve >/dev/null 2>%t
; RUN: FileCheck %s < %t

; CHECK: Passing consecutive scalable vector registers unsupported

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define float @foo(double* %x0, double* %x1) {
entry:
  %0 = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %1 = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %0)
  %2 = call <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1(<vscale x 2 x i1> %1, double* %x0)
  %3 = call <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1(<vscale x 2 x i1> %1, double* %x1)
  %call = call float @callee(float 1.000000e+00, <vscale x 8 x double> %2, <vscale x 8 x double> %3)
  ret float %call
}

declare float @callee(float, <vscale x 8 x double>, <vscale x 8 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 immarg)
declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)
declare <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1(<vscale x 2 x i1>, double*)
