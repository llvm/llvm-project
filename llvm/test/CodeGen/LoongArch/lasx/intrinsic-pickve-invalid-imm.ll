; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvpickve.w(<8 x i32>, i32)

define <8 x i32> @lasx_xvpickve_w_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpickve.w(<8 x i32> %va, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvpickve_w_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpickve.w(<8 x i32> %va, i32 8)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvpickve.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvpickve_d_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpickve.d(<4 x i64> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvpickve_d_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpickve.d(<4 x i64> %va, i32 4)
  ret <4 x i64> %res
}

declare <8 x float> @llvm.loongarch.lasx.xvpickve.w.f(<8 x float>, i32)

define <8 x float> @lasx_xvpickve_w_f_lo(<8 x float> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.w.f: argument out of range
entry:
  %res = call <8 x float> @llvm.loongarch.lasx.xvpickve.w.f(<8 x float> %va, i32 -1)
  ret <8 x float> %res
}

define <8 x float> @lasx_xvpickve_w_f_hi(<8 x float> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.w.f: argument out of range
entry:
  %res = call <8 x float> @llvm.loongarch.lasx.xvpickve.w.f(<8 x float> %va, i32 8)
  ret <8 x float> %res
}

declare <4 x double> @llvm.loongarch.lasx.xvpickve.d.f(<4 x double>, i32)

define <4 x double> @lasx_xvpickve_d_f_lo(<4 x double> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.d.f: argument out of range
entry:
  %res = call <4 x double> @llvm.loongarch.lasx.xvpickve.d.f(<4 x double> %va, i32 -1)
  ret <4 x double> %res
}

define <4 x double> @lasx_xvpickve_d_f_hi(<4 x double> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpickve.d.f: argument out of range
entry:
  %res = call <4 x double> @llvm.loongarch.lasx.xvpickve.d.f(<4 x double> %va, i32 4)
  ret <4 x double> %res
}
