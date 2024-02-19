; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvpickve.w(<8 x i32>, i32)

define <8 x i32> @lasx_xvpickve_w(<8 x i32> %va, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpickve.w(<8 x i32> %va, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvpickve.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvpickve_d(<4 x i64> %va, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpickve.d(<4 x i64> %va, i32 %c)
  ret <4 x i64> %res
}

declare <8 x float> @llvm.loongarch.lasx.xvpickve.w.f(<8 x float>, i32)

define <8 x float> @lasx_xvpickve_w_f(<8 x float> %va, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x float> @llvm.loongarch.lasx.xvpickve.w.f(<8 x float> %va, i32 %c)
  ret <8 x float> %res
}

declare <4 x double> @llvm.loongarch.lasx.xvpickve.d.f(<4 x double>, i32)

define <4 x double> @lasx_xvpickve_d_f(<4 x double> %va, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x double> @llvm.loongarch.lasx.xvpickve.d.f(<4 x double> %va, i32 %c)
  ret <4 x double> %res
}
