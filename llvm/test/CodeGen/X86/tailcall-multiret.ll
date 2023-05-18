; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=core2 | FileCheck %s
; See PR19530
declare double    @llvm.powi.f64.i32(double %Val, i32 %power)
define <3 x double> @julia_foo17589(i32 %arg) {
  %tmp1 = call double @llvm.powi.f64.i32(double 1.000000e+00, i32 %arg)
; CHECK: callq   __powidf2
  %tmp2 = insertelement <3 x double> undef, double %tmp1, i32 0
  %tmp3 = call double @llvm.powi.f64.i32(double 2.000000e+00, i32 %arg)
; CHECK: callq   __powidf2
  %tmp4 = insertelement <3 x double> %tmp2, double %tmp3, i32 1
  %tmp5 = call double @llvm.powi.f64.i32(double 3.000000e+00, i32 %arg)
; CHECK: callq   __powidf2
  %tmp6 = insertelement <3 x double> %tmp4, double %tmp5, i32 2
; CHECK-NOT: TAILCALL
  ret <3 x double> %tmp6
}
