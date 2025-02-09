; RUN: llc %s -filetype=obj

target triple = "wasm32-unknown-unknown"

; In .lto_set_conditional, the first symbol is renamed to the second symbol, so
; the first symbol does not exist anymore in the file. Object writer should not
; crash when .lto_set_conditional is present.

module asm ".lto_set_conditional a,a.new"

define hidden void @a.new() {
  ret void
}
