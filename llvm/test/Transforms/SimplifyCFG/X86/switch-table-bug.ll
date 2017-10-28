; RUN: opt -S -simplifycfg -switch-to-lookup < %s -mtriple=x86_64-apple-darwin12.0.0 | FileCheck %s
; rdar://17735071
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin12.0.0"

; When tableindex can't fit into i2, we should extend the type to i3.
; CHECK-LABEL: @_TFO6reduce1E5toRawfS0_FT_Si
; CHECK: entry:
; CHECK-NEXT: sub i2 %0, -2
; CHECK-NEXT: zext i2 %switch.tableidx to i3
; CHECK-NEXT: getelementptr inbounds [4 x i64], [4 x i64]* @switch.table._TFO6reduce1E5toRawfS0_FT_Si, i32 0, i3 %switch.tableidx.zext
; CHECK-NEXT: load i64, i64* %switch.gep
; CHECK-NEXT: ret i64 %switch.load
define i64 @_TFO6reduce1E5toRawfS0_FT_Si(i2) {
entry:
  switch i2 %0, label %1 [
    i2 0, label %2
    i2 1, label %3
    i2 -2, label %4
    i2 -1, label %5
  ]

; <label>:1                                       ; preds = %entry
  unreachable

; <label>:2                                       ; preds = %2
  br label %6

; <label>:3                                       ; preds = %4
  br label %6

; <label>:4                                       ; preds = %6
  br label %6

; <label>:5                                       ; preds = %8
  br label %6

; <label>:6                                      ; preds = %3, %5, %7, %9
  %7 = phi i64 [ 3, %5 ], [ 2, %4 ], [ 1, %3 ], [ 0, %2 ]
  ret i64 %7
}
