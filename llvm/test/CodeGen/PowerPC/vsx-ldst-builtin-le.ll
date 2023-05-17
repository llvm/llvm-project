; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-P9UP

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=-power9-vector -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr10 -O2 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-P9UP

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
; CHECK-P9UP-LABEL: test1
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %0 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr @vsi)
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <4 x i32> %0, ptr @res_vsi, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %1 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr @vui)
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <4 x i32> %1, ptr @res_vui, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %2 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr @vf)
  %3 = bitcast <4 x i32> %2 to <4 x float>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <4 x float> %3, ptr @res_vf, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %4 = call <2 x double> @llvm.ppc.vsx.lxvd2x(ptr @vsll)
  %5 = bitcast <2 x double> %4 to <2 x i64>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <2 x i64> %5, ptr @res_vsll, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %6 = call <2 x double> @llvm.ppc.vsx.lxvd2x(ptr @vull)
  %7 = bitcast <2 x double> %6 to <2 x i64>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <2 x i64> %7, ptr @res_vull, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %8 = call <2 x double> @llvm.ppc.vsx.lxvd2x(ptr @vd)
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  store <2 x double> %8, ptr @res_vd, align 16
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %9 = load <4 x i32>, ptr @vsi, align 16
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %9, ptr @res_vsi)
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %10 = load <4 x i32>, ptr @vui, align 16
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %10, ptr @res_vui)
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %11 = load <4 x float>, ptr @vf, align 16
  %12 = bitcast <4 x float> %11 to <4 x i32>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %12, ptr @res_vf)
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %13 = load <2 x i64>, ptr @vsll, align 16
  %14 = bitcast <2 x i64> %13 to <2 x double>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %14, ptr @res_vsll)
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %15 = load <2 x i64>, ptr @vull, align 16
  %16 = bitcast <2 x i64> %15 to <2 x double>
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %16, ptr @res_vull)
; CHECK: lxvd2x
; CHECK-P9UP-DAG: lxv
  %17 = load <2 x double>, ptr @vd, align 16
; CHECK: stxvd2x
; CHECK-P9UP-DAG: stxv
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %17, ptr @res_vd)
  ret void
}

declare void @llvm.ppc.vsx.stxvd2x(<2 x double>, ptr)
declare void @llvm.ppc.vsx.stxvw4x(<4 x i32>, ptr)
declare <2 x double> @llvm.ppc.vsx.lxvd2x(ptr)
declare <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr)
