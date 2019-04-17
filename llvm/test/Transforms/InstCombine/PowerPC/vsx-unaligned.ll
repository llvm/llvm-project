; Verify that we can create unaligned loads and stores from VSX intrinsics.

; RUN: opt < %s -instcombine -S | FileCheck %s

target triple = "powerpc64-unknown-linux-gnu"

@vf = common global <4 x float> zeroinitializer, align 1
@res_vf = common global <4 x float> zeroinitializer, align 1
@vd = common global <2 x double> zeroinitializer, align 1
@res_vd = common global <2 x double> zeroinitializer, align 1

define void @test1() {
entry:
  %t1 = alloca <4 x float>*, align 8
  %t2 = alloca <2 x double>*, align 8
  store <4 x float>* @vf, <4 x float>** %t1, align 8
  %0 = load <4 x float>*, <4 x float>** %t1, align 8
  %1 = bitcast <4 x float>* %0 to i8*
  %2 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* %1)
  store <4 x float>* @res_vf, <4 x float>** %t1, align 8
  %3 = load <4 x float>*, <4 x float>** %t1, align 8
  %4 = bitcast <4 x float>* %3 to i8*
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %2, i8* %4)
  store <2 x double>* @vd, <2 x double>** %t2, align 8
  %5 = load <2 x double>*, <2 x double>** %t2, align 8
  %6 = bitcast <2 x double>* %5 to i8*
  %7 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* %6)
  store <2 x double>* @res_vd, <2 x double>** %t2, align 8
  %8 = load <2 x double>*, <2 x double>** %t2, align 8
  %9 = bitcast <2 x double>* %8 to i8*
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %7, i8* %9)
  ret void
}

; CHECK-LABEL: @test1
; CHECK: %0 = load <4 x i32>, <4 x i32>* bitcast (<4 x float>* @vf to <4 x i32>*), align 1
; CHECK: store <4 x i32> %0, <4 x i32>* bitcast (<4 x float>* @res_vf to <4 x i32>*), align 1
; CHECK: %1 = load <2 x double>, <2 x double>* @vd, align 1
; CHECK: store <2 x double> %1, <2 x double>* @res_vd, align 1

declare <4 x i32> @llvm.ppc.vsx.lxvw4x(i8*)
declare void @llvm.ppc.vsx.stxvw4x(<4 x i32>, i8*)
declare <2 x double> @llvm.ppc.vsx.lxvd2x(i8*)
declare void @llvm.ppc.vsx.stxvd2x(<2 x double>, i8*)
