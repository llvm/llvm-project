; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define {i32, i32} @test({i32, i32} %a, i1 %c) {
  ; CHECK: %[[#AL:]] = load { i[[#SBITS]], i[[#SBITS]] }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK: %[[#AL0:]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } %[[#AL]], i[[#SBITS]] 0, 0
  ; CHECK: %[[#AL1:]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } %[[#AL]], i[[#SBITS]] 0, 1
  ; CHECK: %[[#PL:]] = phi { i[[#SBITS]], i[[#SBITS]] } [ %[[#AL0]], %T ], [ %[[#AL1]], %F ]
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } %[[#PL]], ptr @__dfsan_retval_tls, align [[ALIGN]]

entry:
  br i1 %c, label %T, label %F
  
T:
  %at = insertvalue {i32, i32} %a, i32 1, 0
  br label %done
  
F:
  %af = insertvalue {i32, i32} %a, i32 1, 1
  br label %done
  
done:
  %b = phi {i32, i32} [%at, %T], [%af, %F]
  ret {i32, i32} %b  
}
