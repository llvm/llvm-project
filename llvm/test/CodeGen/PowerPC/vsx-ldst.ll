; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -O2 \
; RUN:   -mtriple=powerpc64-unknown-linux-gnu < %s > %t
; RUN: grep lxvw4x < %t | count 3
; RUN: grep lxvd2x < %t | count 3
; RUN: grep stxvw4x < %t | count 3
; RUN: grep stxvd2x < %t | count 3

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -O0 -fast-isel=1 \
; RUN:   -mtriple=powerpc64-unknown-linux-gnu < %s > %t
; RUN: grep lxvw4x < %t | count 3
; RUN: grep lxvd2x < %t | count 3
; RUN: grep stxvw4x < %t | count 3
; RUN: grep stxvd2x < %t | count 3

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s > %t
; RUN: grep lxvd2x < %t | count 6
; RUN: grep stxvd2x < %t | count 6

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s > %t
; RUN: grep lxv < %t | count 6
; RUN: grep stxv < %t | count 6


@vsi = global <4 x i32> <i32 -1, i32 2, i32 -3, i32 4>, align 16
@vui = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@vf = global <4 x float> <float -1.500000e+00, float 2.500000e+00, float -3.500000e+00, float 4.500000e+00>, align 16
@vsll = global <2 x i64> <i64 255, i64 -937>, align 16
@vull = global <2 x i64> <i64 1447, i64 2894>, align 16
@vd = global <2 x double> <double 3.500000e+00, double -7.500000e+00>, align 16
@res_vsi = common global <4 x i32> zeroinitializer, align 16
@res_vui = common global <4 x i32> zeroinitializer, align 16
@res_vf = common global <4 x float> zeroinitializer, align 16
@res_vsll = common global <2 x i64> zeroinitializer, align 16
@res_vull = common global <2 x i64> zeroinitializer, align 16
@res_vd = common global <2 x double> zeroinitializer, align 16

; Function Attrs: nounwind
define void @test1() {
entry:
  %0 = load <4 x i32>, ptr @vsi, align 16
  %1 = load <4 x i32>, ptr @vui, align 16
  %2 = load <4 x i32>, ptr @vf, align 16
  %3 = load <2 x double>, ptr @vsll, align 16
  %4 = load <2 x double>, ptr @vull, align 16
  %5 = load <2 x double>, ptr @vd, align 16
  store <4 x i32> %0, ptr @res_vsi, align 16
  store <4 x i32> %1, ptr @res_vui, align 16
  store <4 x i32> %2, ptr @res_vf, align 16
  store <2 x double> %3, ptr @res_vsll, align 16
  store <2 x double> %4, ptr @res_vull, align 16
  store <2 x double> %5, ptr @res_vd, align 16
  ret void
}
