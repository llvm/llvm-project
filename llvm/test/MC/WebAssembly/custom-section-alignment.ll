; RUN: llc -filetype=obj %s -o - | od -t x1 -v | FileCheck %s

target triple = "wasm32-unknown-unknown"

!0 = !{ !"before", !"\de\ad\be\ef" }
!1 = !{ !"__clangast", !"\fe\ed\fa\ce" }
!wasm.custom_sections = !{ !0, !1 }

; Ensure that __clangast content is aligned by 4 bytes
; CHECK: {{(([0-9a-f]{2} ){4})*}}fe ed fa ce
