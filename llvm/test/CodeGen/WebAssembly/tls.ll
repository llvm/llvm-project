; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck --check-prefix=SINGLE %s
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; SINGLE-LABEL: address_of_tls:
define i32 @address_of_tls() {
  ; SINGLE: i32.const $push0=, tls
  ; SINGLE-NEXT: return $pop0
  ret i32 ptrtoint(i32* @tls to i32)
}

; SINGLE: .type	tls,@object
; SINGLE-NEXT: .section	.bss.tls,"",@
; SINGLE-NEXT: .p2align 2
; SINGLE-NEXT: tls:
; SINGLE-NEXT: .int32 0
@tls = internal thread_local global i32 0
