; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+multivalue | FileCheck %s

; Test that the multivalue attribute is accepted
; TODO(tlively): implement multivalue

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%pair = type { i32, i32 }
%packed_pair = type <{ i32, i32 }>

; CHECK-LABEL: sret:
; CHECK-NEXT: sret (i32, i32, i32) -> ()
define %pair @sret(%pair %p) {
  ret %pair %p
}

; CHECK-LABEL: packed_sret:
; CHECK-NEXT: packed_sret (i32, i32, i32) -> ()
define %packed_pair @packed_sret(%packed_pair %p) {
  ret %packed_pair %p
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 10
; CHECK-NEXT: .ascii "multivalue"
