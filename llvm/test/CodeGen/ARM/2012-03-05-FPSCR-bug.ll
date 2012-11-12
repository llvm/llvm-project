; RUN: llc -march=arm -mcpu=cortex-a8 -verify-machineinstrs < %s
; PR12165
target datalayout = "e-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-p:32:32:32-v128:32:32"
target triple = "arm-none-linux"

define hidden void @_strtod_r() nounwind {
  br i1 undef, label %1, label %2

; <label>:1                                       ; preds = %0
  br label %2

; <label>:2                                       ; preds = %1, %0
  br i1 undef, label %3, label %8

; <label>:3                                       ; preds = %2
  br i1 undef, label %4, label %7

; <label>:4                                       ; preds = %3
  %5 = call i32 @llvm.flt.rounds()
  %6 = icmp eq i32 %5, 1
  br i1 %6, label %8, label %7

; <label>:7                                       ; preds = %4, %3
  unreachable

; <label>:8                                       ; preds = %4, %2
  br i1 undef, label %9, label %10

; <label>:9                                       ; preds = %8
  br label %10

; <label>:10                                      ; preds = %9, %8
  ret void
}

declare i32 @llvm.flt.rounds() nounwind
