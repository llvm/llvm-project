; Test that the strcpy library call simplifier works correctly for ARM procedure calls
; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@hello = constant [6 x i8] c"hello\00"
@a = common global [32 x i8] zeroinitializer, align 1
@b = common global [32 x i8] zeroinitializer, align 1

declare i8* @strcpy(i8*, i8*)

define arm_aapcscc void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0

  call arm_aapcscc i8* @strcpy(i8* %dst, i8* %src)
; CHECK: @llvm.memcpy.p0i8.p0i8.i32
  ret void
}

define arm_aapcscc i8* @test_simplify2() {
; CHECK-LABEL: @test_simplify2(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0

  %ret = call arm_aapcscc i8* @strcpy(i8* %dst, i8* %dst)
; CHECK: ret i8* getelementptr inbounds ([32 x i8], [32 x i8]* @a, i32 0, i32 0)
  ret i8* %ret
}

define arm_aapcscc i8* @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [32 x i8], [32 x i8]* @b, i32 0, i32 0

  %ret = call arm_aapcscc i8* @strcpy(i8* %dst, i8* %src)
; CHECK: call arm_aapcscc i8* @strcpy
  ret i8* %ret
}

define arm_aapcs_vfpcc void @test_simplify1_vfp() {
; CHECK-LABEL: @test_simplify1_vfp(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0

  call arm_aapcs_vfpcc i8* @strcpy(i8* %dst, i8* %src)
; CHECK: @llvm.memcpy.p0i8.p0i8.i32
  ret void
}

define arm_aapcs_vfpcc i8* @test_simplify2_vfp() {
; CHECK-LABEL: @test_simplify2_vfp(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0

  %ret = call arm_aapcs_vfpcc i8* @strcpy(i8* %dst, i8* %dst)
; CHECK: ret i8* getelementptr inbounds ([32 x i8], [32 x i8]* @a, i32 0, i32 0)
  ret i8* %ret
}

define arm_aapcs_vfpcc i8* @test_no_simplify1_vfp() {
; CHECK-LABEL: @test_no_simplify1_vfp(

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [32 x i8], [32 x i8]* @b, i32 0, i32 0

  %ret = call arm_aapcs_vfpcc i8* @strcpy(i8* %dst, i8* %src)
; CHECK: call arm_aapcs_vfpcc i8* @strcpy
  ret i8* %ret
}
