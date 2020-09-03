; RUN: llc < %s -mattr=+reference-types | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: reference-types
define void @reference-types() {
  ret void
}

; CHECK:      .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 15
; CHECK-NEXT: .ascii "reference-types"
