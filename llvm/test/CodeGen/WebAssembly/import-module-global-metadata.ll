; RUN: llc < %s -asm-verbose=false | FileCheck --check-prefix=ASM %s
; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

target triple = "wasm32-unknown-unknown"

@imported_g = external addrspace(1) global i32, !wasm.import.module !0

define i32 @get() {
  %v = load i32, ptr addrspace(1) @imported_g
  ret i32 %v
}

; ASM: .globaltype imported_g, i32
; ASM-NEXT: .import_module imported_g, "js"

; OBJ:      - Type:            IMPORT
; OBJ:        Imports:
; OBJ:          - Module:          js
; OBJ-NEXT:       Field:           imported_g
; OBJ-NEXT:       Kind:            GLOBAL
; OBJ-NEXT:       GlobalType:      I32
; OBJ-NEXT:       GlobalMutable:   true

!0 = !{!"js"}