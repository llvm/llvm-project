; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+tail-call | FileCheck %s

; Test that the tail-call attribute is accepted
; TODO(tlively): implement tail call

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: recursive_tail:
; CHECK:      i32.call $push[[L0:[0-9]+]]=, recursive_tail{{$}}
; CHECK-NEXT: return $pop[[L0]]{{$}}
define i32 @recursive_tail() {
  %v = tail call i32 @recursive_tail()
  ret i32 %v
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 9
; CHECK-NEXT: .ascii "tail-call"
