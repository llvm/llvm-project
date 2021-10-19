; RUN: llc < %s -fast-isel --mtriple=wasm64 -asm-verbose=false -wasm-keep-registers | FileCheck %s

target triple = "wasm64"

; Ensure fast isel also lowers function pointers to 32-bit.

; CHECK:       local.get $push[[L0:[0-9]+]]=, 0
; CHECK-NEXT:  i32.wrap_i64 $push[[L1:[0-9]+]]=, $pop[[L0]]
; CHECK-NEXT:  call_indirect $pop[[L1]]

define hidden void @f(void ()* %g) {
  call void %g()
  ret void
}
