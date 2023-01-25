; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown    | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

declare void @bar(i64)

@b = global i16 0, align 2

; This test case is extracted from PR38125.
; %or is reachable by both a sext and zext that are going to be promoted.
; It ensures correct operation on PromotedInsts.

; CHECK:       %promoted = trunc i32 %or to i16
; CHECK-NEXT:  %c = sext i16 %promoted to i64
define i32 @foo(i16 %kkk) {
entry:
  %t4 = load i16, ptr @b, align 2
  %conv4 = zext i16 %t4 to i32
  %or = or i16 %kkk, %t4
  %c = sext i16 %or to i64
  call void @bar(i64 %c)
  %t5 = and i16 %or, 5
  %z = zext i16 %t5 to i32
  ret i32 %z
}
