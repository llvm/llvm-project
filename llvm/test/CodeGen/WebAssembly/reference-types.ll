; RUN: llc < %s -mcpu=mvp -mattr=+reference-types | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: reference-types
define void @reference-types() {
  ret void
}

; CHECK: .section .custom_section.target_features,"",@
; CHECK-NEXT: .int8 2
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 22
; CHECK-NEXT: .ascii "call-indirect-overlong"
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 15
; CHECK-NEXT: .ascii "reference-types"
