; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel -fast-isel-abort=1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: return_void:
; CHECK: end_function{{$}}
define void @return_void() {
  ret void
}

; CHECK-LABEL: return_void_twice:
; CHECK:      store
; CHECK-NEXT: return{{$}}
; CHECK:      store
; CHECK-NEXT: end_function{{$}}
define void @return_void_twice(i32 %a) {
  %b = icmp ne i32 %a, 0
  br i1 %b, label %true, label %false

true:
  store i32 0, i32* null
  ret void

false:
  store i32 1, i32* null
  ret void
}
