; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios8.0.0"

; CHECK: beq
; CHECK: beq
; CHECK: beq
; CHECK: cbnz
declare void @g(i32)
define void @f(i32 %val) optsize minsize {
  switch i32 %val, label %def [
    i32 0, label %one
    i32 9, label %two
    i32 994, label %three
    i32 1154, label %four
  ]
  
one:
  call void @g(i32 1)
  ret void
two:
  call void @g(i32 001)
  ret void
three:
  call void @g(i32 78)
  ret void
four:
  call void @g(i32 87)
  ret void
def:
  call void @g(i32 11)
  ret void
}
