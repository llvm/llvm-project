; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

; Test that basic functions assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: f0:
; CHECK: return{{$}}
; CHECK: end_function{{$}}
; CHECK: .size f0,
define void @f0() {
  ret void
}

; CHECK-LABEL: f1:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
; CHECK: .size f1,
define i32 @f1() {
  ret i32 0
}

; CHECK-LABEL: f2:
; CHECK-NEXT: .param i32, f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
; CHECK: .size f2,
define i32 @f2(i32 %p1, float %p2) {
  ret i32 0
}

; CHECK-LABEL: f3:
; CHECK-NEXT: .param i32, f32{{$}}
; CHECK-NOT: local
; CHECK-NEXT: return{{$}}
; CHECK: .size f3,
define void @f3(i32 %p1, float %p2) {
  ret void
}

; CHECK-LABEL: f4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NOT: local
; CHECK: .size f4,
define i32 @f4(i32 %x) {
entry:
   %c = trunc i32 %x to i1
   br i1 %c, label %true, label %false
true:
   ret i32 0
false:
   ret i32 1
}

; CHECK-LABEL: f5:
; CHECK-NEXT: .result f32{{$}}
; CHECK-NEXT: unreachable
define float @f5()  {
 unreachable
}
