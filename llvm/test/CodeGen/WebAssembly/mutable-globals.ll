; RUN: llc < %s -mattr=+mutable-globals | FileCheck %s

; Test that mutable globals is properly emitted into the target features section

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @foo() {
  ret void
}

; CHECK-LABEL: .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 15
; CHECK-NEXT: .ascii "mutable-globals"
