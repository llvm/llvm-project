; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Function that uses explict stack, and should generate a reference to
; __stack_pointer, along with the corresponding reloction entry.
define hidden void @foo() #0 {
entry:
  alloca i32, align 4
  ret void
}

; CHECK:  - Type:            IMPORT
; CHECK:     Imports:
; CHECK:       - Module:          env
; CHECK:         Field:           __stack_pointer
; CHECK:         Kind:            GLOBAL
; CHECK:         GlobalType:      I32
; CHECK:         GlobalMutable:   true
; CHECK:   - Type:            CODE
; CHECK:     Relocations:
; CHECK:       - Type:            R_WASM_GLOBAL_INDEX_LEB
; CHECK:         Index:           0
