; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Make sure that argument offsets are correct even if some arguments are unused.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: unused_first:
; CHECK-NEXT: .functype unused_first (i32, i32) -> (i32){{$}}
; CHECK-NEXT: return $1{{$}}
define i32 @unused_first(i32 %x, i32 %y) {
  ret i32 %y
}

; CHECK-LABEL: unused_second:
; CHECK-NEXT: .functype unused_second (i32, i32) -> (i32){{$}}
; CHECK-NEXT: return $0{{$}}
define i32 @unused_second(i32 %x, i32 %y) {
  ret i32 %x
}

; CHECK-LABEL: call_something:
; CHECK:      {{^}} i32.call $drop=, return_something{{$}}
; CHECK-NEXT: return{{$}}
declare i32 @return_something()
define void @call_something() {
  call i32 @return_something()
  ret void
}
