; Test that the strcpy library call simplifier works correctly for ARM procedure calls
; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@hello = constant [6 x i8] c"hello\00"
@a = common global [32 x i8] zeroinitializer, align 1
@b = common global [32 x i8] zeroinitializer, align 1

declare ptr @strcpy(ptr, ptr)

define arm_aapcscc void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(


  call arm_aapcscc ptr @strcpy(ptr @a, ptr @hello)
; CHECK: @llvm.memcpy.p0.p0.i32
  ret void
}

define arm_aapcscc ptr @test_simplify2() {
; CHECK-LABEL: @test_simplify2(


  %ret = call arm_aapcscc ptr @strcpy(ptr @a, ptr @a)
; CHECK: ret ptr @a
  ret ptr %ret
}

define arm_aapcscc ptr @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(


  %ret = call arm_aapcscc ptr @strcpy(ptr @a, ptr @b)
; CHECK: call arm_aapcscc ptr @strcpy
  ret ptr %ret
}

define arm_aapcs_vfpcc void @test_simplify1_vfp() {
; CHECK-LABEL: @test_simplify1_vfp(


  call arm_aapcs_vfpcc ptr @strcpy(ptr @a, ptr @hello)
; CHECK: @llvm.memcpy.p0.p0.i32
  ret void
}

define arm_aapcs_vfpcc ptr @test_simplify2_vfp() {
; CHECK-LABEL: @test_simplify2_vfp(


  %ret = call arm_aapcs_vfpcc ptr @strcpy(ptr @a, ptr @a)
; CHECK: ret ptr @a
  ret ptr %ret
}

define arm_aapcs_vfpcc ptr @test_no_simplify1_vfp() {
; CHECK-LABEL: @test_no_simplify1_vfp(


  %ret = call arm_aapcs_vfpcc ptr @strcpy(ptr @a, ptr @b)
; CHECK: call arm_aapcs_vfpcc ptr @strcpy
  ret ptr %ret
}
