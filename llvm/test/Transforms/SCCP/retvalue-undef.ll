; RUN: opt -ipsccp -S %s | FileCheck %s
; PR6414
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 ()* @f() {
  ret i32 ()* @g
}

define internal i32 @g() {
  ret i32 8
}

; CHECK: internal i32 @g()
; CHECK-NEXT: ret i32 8

define internal void @outer_mod() {
  %1 = call i32 ()* ()* @f()                      ; <i32 ()*> [#uses=1]
  %2 = call i32 %1()                              ; <i32> [#uses=0]
  ret void
}

define internal void @module_init() {
  call void @register_outer_mod(void ()* @outer_mod)
  ret void
}

declare void @register_outer_mod(void ()*)

define i32 @main() {
  ret i32 0
}
