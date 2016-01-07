; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: return_void:
; CHECK: return{{$}}
define void @return_void() {
  ret void
}
