; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: return_i32:
; CHECK: return $0{{$}}
define i32 @return_i32(i32 %p) {
  ret i32 %p
}
