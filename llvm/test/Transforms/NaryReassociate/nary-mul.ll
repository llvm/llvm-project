; RUN: opt < %s -nary-reassociate -S | FileCheck %s
; RUN: opt < %s -passes='nary-reassociate' -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

declare void @foo(i32)

; CHECK-LABEL: @bar(
define void @bar(i32 %a, i32 %b, i32 %c) {
  %1 = mul i32 %a, %c
; CHECK: [[BASE:%[a-zA-Z0-9]+]] = mul i32 %a, %c
  call void @foo(i32 %1)
  %2 = mul i32 %a, %b
  %3 = mul i32 %2, %c
; CHECK: [[RESULT:%[a-zA-Z0-9]+]] = mul i32 [[BASE]], %b
  call void @foo(i32 %3)
; CHECK-NEXT: call void @foo(i32 [[RESULT]])
  ret void
}

