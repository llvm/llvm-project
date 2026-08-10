; Verify that we can create unaligned loads and stores from VSX intrinsics.

; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target triple = "powerpc64-unknown-linux-gnu"

@vf = common global <4 x float> zeroinitializer, align 1
@res_vf = common global <4 x float> zeroinitializer, align 1
@vd = common global <2 x double> zeroinitializer, align 1
@res_vd = common global <2 x double> zeroinitializer, align 1

define void @test1() {
entry:
  %t1 = alloca ptr, align 8
  %t2 = alloca ptr, align 8
  store ptr @vf, ptr %t1, align 8
  %0 = load ptr, ptr %t1, align 8
  %1 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %0)
  store ptr @res_vf, ptr %t1, align 8
  %2 = load ptr, ptr %t1, align 8
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %1, ptr %2)
  store ptr @vd, ptr %t2, align 8
  %3 = load ptr, ptr %t2, align 8
  %4 = call <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %3)
  store ptr @res_vd, ptr %t2, align 8
  %5 = load ptr, ptr %t2, align 8
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %4, ptr %5)
  ret void
}

; CHECK-LABEL: @test1
; CHECK: %0 = load <4 x i32>, ptr @vf, align 1
; CHECK: store <4 x i32> %0, ptr @res_vf, align 1
; CHECK: %1 = load <2 x double>, ptr @vd, align 1
; CHECK: store <2 x double> %1, ptr @res_vd, align 1

declare <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr)
declare void @llvm.ppc.vsx.stxvw4x(<4 x i32>, ptr)
declare <2 x double> @llvm.ppc.vsx.lxvd2x(ptr)
declare void @llvm.ppc.vsx.stxvd2x(<2 x double>, ptr)
