; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test the mechanism for defining user custom sections.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

!0 = !{ !"red", !"foo" }
!1 = !{ !"green", !"bar" }
!2 = !{ !"green", !"qux" }
!wasm.custom_sections = !{ !0, !1, !2 }

; CHECK:      .section	.custom_section.red,"",@
; CHECK-NEXT: .ascii	"foo"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"bar"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"qux"
