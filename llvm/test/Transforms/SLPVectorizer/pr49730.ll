; RUN: opt -slp-vectorizer -S < %s | FileCheck %s
; RUN: opt -passes=slp-vectorizer -S < %s | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; This should not crash.
define void @test() {

; CHECK-LABEL: test

bb:
  %tmp = call i32 @llvm.smin.i32(i32 undef, i32 2)
  %tmp1 = sub nsw i32 undef, %tmp
  %tmp2 = call i32 @llvm.umin.i32(i32 undef, i32 %tmp1)
  %tmp3 = call i32 @llvm.smin.i32(i32 undef, i32 2)
  %tmp4 = sub nsw i32 undef, %tmp3
  %tmp5 = call i32 @llvm.umin.i32(i32 %tmp2, i32 %tmp4)
  %tmp6 = call i32 @llvm.smin.i32(i32 undef, i32 1)
  %tmp7 = sub nuw nsw i32 undef, %tmp6
  %tmp8 = call i32 @llvm.umin.i32(i32 %tmp5, i32 %tmp7)
  %tmp9 = call i32 @llvm.smin.i32(i32 undef, i32 1)
  %tmp10 = sub nsw i32 undef, %tmp9
  %tmp11 = call i32 @llvm.umin.i32(i32 %tmp8, i32 %tmp10)
  %tmp12 = sub nsw i32 undef, undef
  %tmp13 = call i32 @llvm.umin.i32(i32 %tmp11, i32 %tmp12)
  %tmp14 = call i32 @llvm.umin.i32(i32 %tmp13, i32 93)
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smin.i32(i32, i32)

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.umin.i32(i32, i32)
