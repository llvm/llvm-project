; RUN: llc < %s -asm-verbose=false -fast-isel -fast-isel-abort=1 -verify-machineinstrs | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare void @extern48(i48)

; CHECK-LABEL: call_trunc_i64_to_i48:
; CHECK:       local.get 0
; CHECK-NEXT:  call extern48
; CHECK-NEXT:  end_function
define void @call_trunc_i64_to_i48(i64 %x) {
  %x48 = trunc i64 %x to i48
  call void @extern48(i48 %x48)
  ret void
}
