; RUN: llc < %s | FileCheck %s

; Test that wasm64 is properly emitted into the target features section

target triple = "wasm64-unknown-unknown"

define void @foo() {
  ret void
}

; CHECK-LABEL: .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 8
; CHECK-NEXT: .ascii "memory64"
