; RUN: llc < %s -fast-isel -mcpu=core2 -mtriple=x86_64-unknown-unknown -O1 | FileCheck %s
; See PR21557

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @bar(i1)

define i64 @foo(ptr %arg) {
; CHECK-LABEL: foo:
top:
  %0 = load i8, ptr %arg
; CHECK: movzbl
  %1 = trunc i8 %0 to i1
; CHECK: andb $1,
  %2 = call i64 @bar(i1 %1)
; CHECK: callq
  ret i64 %2
}
