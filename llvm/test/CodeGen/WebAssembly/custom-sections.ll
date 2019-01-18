; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test the mechanism for defining user custom sections.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

!0 = !{ !"red", !"foo" }
!1 = !{ !"green", !"bar" }
!2 = !{ !"green", !"qux" }
!wasm.custom_sections = !{ !0, !1, !2 }

!llvm.ident = !{!3}
!3 = !{!"clang version 123"}

; CHECK:      .section	.custom_section.red,"",@
; CHECK-NEXT: .ascii	"foo"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"bar"

; CHECK:      .section	.custom_section.green,"",@
; CHECK-NEXT: .ascii	"qux"

; CHECK:      .section	.custom_section.producers,"",@
; CHECK-NEXT: .int8	1
; CHECK-NEXT: .int8	12
; CHECK-NEXT: .ascii	"processed-by"
; CHECK-NEXT: .int8	1
; CHECK-NEXT: .int8	5
; CHECK-NEXT: .ascii	"clang"
; CHECK-NEXT: .int8	3
; CHECK-NEXT: .ascii	"123"
