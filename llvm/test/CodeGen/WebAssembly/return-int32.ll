; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel -fast-isel-abort=1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: return_i32:
; CHECK-NEXT:  .param i32{{$}}
; CHECK-NEXT:  .result i32{{$}}
; CHECK-NEXT:  get_local  $push0=, 0
; CHECK-NEXT:  end_function{{$}}
define i32 @return_i32(i32 %p) {
  ret i32 %p
}

; CHECK-LABEL: return_i32_twice:
; CHECK:      store
; CHECK-NEXT: i32.const $push[[L0:[^,]+]]=, 1{{$}}
; CHECK-NEXT: return $pop[[L0]]{{$}}
; CHECK:      store
; CHECK-NEXT: i32.const $push{{[^,]+}}=, 3{{$}}
; CHECK-NEXT: end_function{{$}}
define i32 @return_i32_twice(i32 %a) {
  %b = icmp ne i32 %a, 0
  br i1 %b, label %true, label %false

true:
  store i32 0, i32* null
  ret i32 1 

false:
  store i32 2, i32* null
  ret i32 3
}
