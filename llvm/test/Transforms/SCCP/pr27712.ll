; RUN: opt -passes=sccp -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  br label %lbl_1154

lbl_1154:
  %b0.0 = phi i32 [ -119, %entry ], [ 0, %lbl_1154 ]
  %cmp11 = icmp slt i32 %b0.0, 0
  %shl.op = shl i32 33554432, %b0.0
  %cmp1445 = icmp ult i32 %shl.op, 33554432
  %cmp14 = or i1 %cmp11, %cmp1445
  br i1 %cmp14, label %lbl_1154, label %if.end19

if.end19:
  br i1 %cmp11, label %if.then22, label %cleanup26

if.then22:
  tail call void @abort()
  unreachable

cleanup26:
  ret i32 %shl.op
}
; CHECK-LABEL: define i32 @main(
; CHECK-NOT: ret i32 undef

declare void @abort()
