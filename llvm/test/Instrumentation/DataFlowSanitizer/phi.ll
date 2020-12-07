; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define {i32, i32} @test({i32, i32} %a, i1 %c) {
  ; CHECK: [[E0:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; CHECK: [[E3:%.*]] = phi i16 [ [[E0]], %T ], [ [[E0]], %F ]
  ; CHECK: store i16 [[E3]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

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
