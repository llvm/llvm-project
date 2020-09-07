; RUN: opt -simplifycfg -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g()
declare void @f()

define void @foo(i32 %Kind) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:entry:
; CHECK-NEXT:  switch i32 %Kind, label %sw.epilog [
; CHECK-NEXT:    i32 15, label %sw.bb2
; CHECK-NEXT:    i32 2, label %sw.bb
; CHECK-NEXT:  ]
; CHECK:     sw.bb:
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  br label %sw.epilog
; CHECK:     sw.bb2:
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  br label %sw.epilog
; CHECK:     sw.epilog:
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

entry:
  switch i32 %Kind, label %sw.epilog [
    i32 1, label %sw.epilog
    i32 2, label %sw.bb
    i32 15, label %sw.bb2
  ]

sw.bb:
  call void @g()
  call void @g()
  br label %sw.epilog

sw.bb2:
  call void @f()
  br label %sw.epilog

sw.epilog:
  ret void
}
