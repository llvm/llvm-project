; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic functions assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: f0:
; CHECK: return{{$}}
; CHECK: .size f0,
define void @f0() {
  ret void
}

; CHECK-LABEL: f1:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: i32.const $push, 0{{$}}
; CHECK-NEXT: set_local 0, $pop{{$}}
; CHECK-NEXT: return (get_local 0){{$}}
; CHECK: .size f1,
define i32 @f1() {
  ret i32 0
}

; CHECK-LABEL: f2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: i32.const $push, 0{{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
; CHECK: .size f2,
define i32 @f2(i32 %p1, float %p2) {
  ret i32 0
}

; CHECK-LABEL: f3:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NOT: .local
; CHECK-NEXT: return{{$}}
; CHECK: .size f3,
define void @f3(i32 %p1, float %p2) {
  ret void
}

; CHECK-LABEL: f4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local
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
