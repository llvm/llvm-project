; Verify that we don't add incorrect llvm.loop metadata to a
; non-latching branch.

; RUN: opt -passes=callsite-splitting -S < %s 2>&1 | FileCheck %s

; CHECK-NOT: br label %x{{.*}}, !llvm.loop !0

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @foo(i1 %0, i1 %1, i1 %2) {
  br i1 %0, label %x1, label %x2

x1:
  br label %x16

x2:
  br label %x16

x16:
  %x17 = phi i32 [ 0, %x1 ], [ 1, %x2 ]
  br i1 %1, label %x55, label %x46

x46:
  %x47 = icmp eq i32 %x17, 0
  br i1 %x47, label %x49, label %x48

x48:
  br i1 %2, label %x49, label %x48, !llvm.loop !0

x49:
  %x50 = tail call fastcc i32 @func1(i32 noundef %x17)
  br label %x55

x55:
  ret void
}

define internal fastcc i32 @func1(i32 noundef %0) unnamed_addr {
  ret i32 13
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
