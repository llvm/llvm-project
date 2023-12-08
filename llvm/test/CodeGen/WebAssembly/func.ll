; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that basic functions assemble as expected.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: f0:
; CHECK: return{{$}}
; CHECK: end_function{{$}}
define void @f0() {
  ret void
}

; CHECK-LABEL: f1:
; CHECK-NEXT: .functype f1 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
; CHECK: end_function{{$}}
define i32 @f1() {
  ret i32 0
}

; CHECK-LABEL: f2:
; CHECK-NEXT: .functype f2 (i32, f32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push[[NUM:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
; CHECK: end_function{{$}}
define i32 @f2(i32 %p1, float %p2) {
  ret i32 0
}

; CHECK-LABEL: f3:
; CHECK-NEXT: .functype f3 (i32, f32) -> (){{$}}
; CHECK-NOT: local
; CHECK-NEXT: return{{$}}
; CHECK: end_function{{$}}
define void @f3(i32 %p1, float %p2) {
  ret void
}

; CHECK-LABEL: f4:
; CHECK-NEXT: .functype f4 (i32) -> (i32){{$}}
; CHECK-NOT: local
; CHECK: end_function{{$}}
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
; CHECK-NEXT: .functype f5 () -> (f32){{$}}
; CHECK-NEXT: unreachable
define float @f5()  {
 unreachable
}
