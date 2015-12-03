; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that the wasm-store-results pass makes users of stored values use the
; result of store expressions to reduce get_local/set_local traffic.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: single_block:
; CHECK-NOT: .local
; CHECK: i32.const $push{{[0-9]+}}=, 0
; CHECK: i32.store $push[[STORE:[0-9]+]]=, $0, $pop{{[0-9]+}}
; CHECK: return $pop[[STORE]]{{$}}
define i32 @single_block(i32* %p) {
entry:
  store i32 0, i32* %p
  ret i32 0
}
