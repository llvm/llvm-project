; TODO(sbc): Make this test pass by adding support for unnamed tempoaries
; in wasm relocations.
; RUN: not llc -filetype=obj %s

target triple = "wasm32-unknown-unknown-wasm"

@foo = internal global i8* blockaddress(@bar, %addr), align 4

define hidden i32 @bar() #0 {
entry:
  br label %addr

addr:
  ret i32 0
}
