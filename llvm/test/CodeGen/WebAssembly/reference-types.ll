; RUN: llc < %s -mattr=+reference-types | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: reference-types
define void @reference-types() {
  ret void
}

; CHECK:      .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 15
; CHECK-NEXT: .ascii "reference-types"
