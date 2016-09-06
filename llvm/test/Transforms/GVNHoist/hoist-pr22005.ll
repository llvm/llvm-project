; RUN: opt -gvn-hoist -S < %s | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that all "sub" expressions are hoisted.
; CHECK-LABEL: @fun
; CHECK: sub i64
; CHECK-NOT: sub i64

define i64 @fun(i8* %out, i8* %end) {
  %1 = icmp ult i8* %out, %end
  br i1 %1, label %2, label %6

; <label>:2                                       ; preds = %0
  %3 = ptrtoint i8* %end to i64
  %4 = ptrtoint i8* %out to i64
  %5 = sub i64 %3, %4
  br label %10

; <label>:6                                       ; preds = %0
  %7 = ptrtoint i8* %out to i64
  %8 = ptrtoint i8* %end to i64
  %9 = sub i64 %8, %7
  br label %10

; <label>:10                                      ; preds = %6, %2
  %.in = phi i64 [ %5, %2 ], [ %9, %6 ]
  %11 = add i64 %.in, 257
  ret i64 %11
}
