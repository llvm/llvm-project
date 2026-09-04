; RUN: opt < %s -passes=dfsan,dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @add(i8 %a, i8 %b) {
  ; CHECK: @add.dfsan
  ; CHECK-DAG: %[[#ALABEL:]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK-DAG: %[[#BLABEL:]] = load i8, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN]]
  ; CHECK: %[[#UNION:]] = or i8 %[[#ALABEL]], %[[#BLABEL]]
  ; CHECK: %c = add i8 %a, %b
  ; CHECK: store i8 %[[#UNION]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK: ret i8 %c
  %c = add i8 %a, %b
  ret i8 %c
}

; CHECK: [[META0:![0-9]+]] = !{i32 4, !"nosanitize_dataflow", i32 1}
