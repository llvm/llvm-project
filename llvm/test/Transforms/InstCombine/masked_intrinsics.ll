; RUN: opt -instcombine -S < %s | FileCheck %s

declare <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

; FIXME: All of these could be simplified.

define <2 x double> @load_zeromask(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 1, <2 x i1> zeroinitializer, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_zeromask(
; CHECK-NEXT:  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 1, <2 x i1> zeroinitializer, <2 x double> %passthru)
; CHECK-NEXT   ret <2 x double> %res
}

define <2 x double> @load_onemask(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 2, <2 x i1> <i1 1, i1 1>, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_onemask(
; CHECK-NEXT:  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 2, <2 x i1> <i1 true, i1 true>, <2 x double> %passthru)
; CHECK-NEXT   ret <2 x double> %res
}

define <2 x double> @load_onesetbitmask1(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 3, <2 x i1> <i1 0, i1 1>, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_onesetbitmask1(
; CHECK-NEXT:  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 3, <2 x i1> <i1 false, i1 true>, <2 x double> %passthru)
; CHECK-NEXT   ret <2 x double> %res
}

define <2 x double> @load_onesetbitmask2(<2 x double>* %ptr, <2 x double> %passthru)  {
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 4, <2 x i1> <i1 1, i1 0>, <2 x double> %passthru)
  ret <2 x double> %res

; CHECK-LABEL: @load_onesetbitmask2(
; CHECK-NEXT:  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %ptr, i32 4, <2 x i1> <i1 true, i1 false>, <2 x double> %passthru)
; CHECK-NEXT   ret <2 x double> %res
}
