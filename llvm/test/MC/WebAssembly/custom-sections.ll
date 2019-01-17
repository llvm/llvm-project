; RUN: llc -filetype=obj %s -o - | llvm-readobj -s | FileCheck %s

; Test the mechanism for defining user custom sections.

target triple = "wasm32-unknown-unknown"

!0 = !{ !"red", !"foo" }
!1 = !{ !"green", !"bar" }
!2 = !{ !"green", !"qux" }
!wasm.custom_sections = !{ !0, !1, !2 }

!3 = !{ !"clang version 123"}
!llvm.ident = !{!3}

; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Size: 3
; CHECK:    Offset: 72
; CHECK:    Name: red
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Size: 6
; CHECK:    Offset: 85
; CHECK:    Name: green
; CHECK:  }
; CHECK:  Section {
; CHECK:    Type: CUSTOM (0x0)
; CHECK:    Size: 25
; CHECK:    Offset: 118
; CHECK:    Name: producers
; CHECK:  }
