; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-P9

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=-power9-vector -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

@vf = global <4 x float> <float -1.500000e+00, float 2.500000e+00, float -3.500000e+00, float 4.500000e+00>, align 16
@vd = global <2 x double> <double 3.500000e+00, double -7.500000e+00>, align 16
@vsi = global <4 x i32> <i32 -1, i32 2, i32 -3, i32 4>, align 16
@vui = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@vsll = global <2 x i64> <i64 255, i64 -937>, align 16
@vull = global <2 x i64> <i64 1447, i64 2894>, align 16
@res_vsi = common global <4 x i32> zeroinitializer, align 16
@res_vui = common global <4 x i32> zeroinitializer, align 16
@res_vf = common global <4 x float> zeroinitializer, align 16
@res_vsll = common global <2 x i64> zeroinitializer, align 16
@res_vull = common global <2 x i64> zeroinitializer, align 16
@res_vd = common global <2 x double> zeroinitializer, align 16

define void @test1() {
entry:
; CHECK-LABEL: test1
; CHECK-P9-LABEL: test1
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %0 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* bitcast (<4 x i32>* @vsi to i8*))
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <4 x i32> %0, <4 x i32>* @res_vsi, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %1 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* bitcast (<4 x i32>* @vui to i8*))
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <4 x i32> %1, <4 x i32>* @res_vui, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %2 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* bitcast (<4 x float>* @vf to i8*))
  %3 = bitcast <4 x i32> %2 to <4 x float>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <4 x float> %3, <4 x float>* @res_vf, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %4 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* bitcast (<2 x i64>* @vsll to i8*))
  %5 = bitcast <2 x double> %4 to <2 x i64>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <2 x i64> %5, <2 x i64>* @res_vsll, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %6 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* bitcast (<2 x i64>* @vull to i8*))
  %7 = bitcast <2 x double> %6 to <2 x i64>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <2 x i64> %7, <2 x i64>* @res_vull, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %8 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* bitcast (<2 x double>* @vd to i8*))
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  store <2 x double> %8, <2 x double>* @res_vd, align 16
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %9 = load <4 x i32>, <4 x i32>* @vsi, align 16
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %9, i8* bitcast (<4 x i32>* @res_vsi to i8*))
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %10 = load <4 x i32>, <4 x i32>* @vui, align 16
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %10, i8* bitcast (<4 x i32>* @res_vui to i8*))
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %11 = load <4 x float>, <4 x float>* @vf, align 16
  %12 = bitcast <4 x float> %11 to <4 x i32>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %12, i8* bitcast (<4 x float>* @res_vf to i8*))
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %13 = load <2 x i64>, <2 x i64>* @vsll, align 16
  %14 = bitcast <2 x i64> %13 to <2 x double>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %14, i8* bitcast (<2 x i64>* @res_vsll to i8*))
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %15 = load <2 x i64>, <2 x i64>* @vull, align 16
  %16 = bitcast <2 x i64> %15 to <2 x double>
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %16, i8* bitcast (<2 x i64>* @res_vull to i8*))
; CHECK: lxvd2x
; CHECK-P9-DAG: lxvx
  %17 = load <2 x double>, <2 x double>* @vd, align 16
; CHECK: stxvd2x
; CHECK-P9-DAG: stxvx
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %17, i8* bitcast (<2 x double>* @res_vd to i8*))
  ret void
}

declare void @llvm.ppc.vsx.stxvd2x(<2 x double>, i8*)
declare void @llvm.ppc.vsx.stxvw4x(<4 x i32>, i8*)
declare <2 x double> @llvm.ppc.vsx.lxvd2x(i8*)
declare <4 x i32> @llvm.ppc.vsx.lxvw4x(i8*)
