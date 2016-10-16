; RUN: opt -gvn %s -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that the loads of @aaa, @bbb and @ccc are hoisted.
; CHECK-LABEL: define void @foo
; CHECK-NEXT:  %2 = load i32, i32* @ccc, align 4
; CHECK-NEXT:  %3 = load i32, i32* @bbb, align 4
; CHECK-NEXT:  %4 = load i32, i32* @aaa, align 4

@aaa = local_unnamed_addr global i32 10, align 4
@bbb = local_unnamed_addr global i32 20, align 4
@ccc = local_unnamed_addr global i32 30, align 4

define void @foo(i32* nocapture readonly) local_unnamed_addr {
  br label %2

  %.0 = phi i32* [ %0, %1 ], [ %3, %22 ]
  %3 = getelementptr inbounds i32, i32* %.0, i64 1
  %4 = load i32, i32* %.0, align 4
  %5 = load i32, i32* @ccc, align 4
  %6 = icmp sgt i32 %4, %5
  br i1 %6, label %7, label %10

  %8 = load i32, i32* @bbb, align 4
  %9 = add nsw i32 %8, %4
  store i32 %9, i32* @bbb, align 4
  br label %10

  %11 = load i32, i32* @bbb, align 4
  %12 = icmp sgt i32 %4, %11
  br i1 %12, label %13, label %16

  %14 = load i32, i32* @aaa, align 4
  %15 = add nsw i32 %14, %4
  store i32 %15, i32* @aaa, align 4
  br label %16

  %17 = load i32, i32* @aaa, align 4
  %18 = icmp sgt i32 %4, %17
  br i1 %18, label %19, label %22

  %20 = load i32, i32* @ccc, align 4
  %21 = add nsw i32 %20, %4
  store i32 %21, i32* @ccc, align 4
  br label %22

  %23 = icmp slt i32 %4, 0
  br i1 %23, label %24, label %2

  ret void
}
