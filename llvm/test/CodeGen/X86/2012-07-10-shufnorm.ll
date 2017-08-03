; RUN: llc < %s -march=x86 -mcpu=corei7 -mattr=+avx | FileCheck %s

; CHECK: ocl
define void @ocl() {
entry:
  %vext = shufflevector <2 x double> zeroinitializer, <2 x double> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit = shufflevector <8 x double> %vext, <8 x double> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit1 = insertelement <8 x double> %vecinit, double undef, i32 2
  %vecinit3 = insertelement <8 x double> %vecinit1, double undef, i32 3
  %vecinit5 = insertelement <8 x double> %vecinit3, double 0.000000e+00, i32 4
  %vecinit9 = shufflevector <8 x double> %vecinit5, <8 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 9, i32 10>
  store <8 x double> %vecinit9, <8 x double>* undef
  ret void
; CHECK: vxorps
; CHECK: ret
}

