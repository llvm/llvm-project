; REQUIRES: asserts
; RUN: opt -loop-reduce -debug-only=loop-reduce -S  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; CHECK: LSR Use: Kind=Address
; CHECK: LSR Use: Kind=Address
; CHECK-NOT: LSR Use: Kind=Basic
; CHECK-NOT: LSR Use: Kind=Basic

declare <4 x i32> @llvm.ppc.altivec.lvx(i8*)
declare void @llvm.ppc.altivec.stvx(<4 x i32>, i8*)

; Function Attrs: nofree norecurse nounwind
define void @foo(<4 x i32>* %0, <4 x i32>* %1, i32 signext %2) {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %7

5:                                                ; preds = %3
  %6 = zext i32 %2 to i64
  br label %8

7:                                                ; preds = %8, %3
  ret void

8:                                                ; preds = %5, %8
  %9 = phi i64 [ 0, %5 ], [ %15, %8 ]
  %10 = getelementptr inbounds <4 x i32>, <4 x i32>* %1, i64 %9
  %11 = bitcast <4 x i32>* %10 to i8*
  %12 = call <4 x i32> @llvm.ppc.altivec.lvx(i8* %11)
  %13 = getelementptr inbounds <4 x i32>, <4 x i32>* %0, i64 %9
  %14 = bitcast <4 x i32>* %13 to i8*
  call void @llvm.ppc.altivec.stvx(<4 x i32> %12, i8* %14)
  %15 = add nuw nsw i64 %9, 10
  %16 = icmp ult i64 %15, %6
  br i1 %16, label %8, label %7
}
