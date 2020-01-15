; TODO(sbc): Make this test pass by adding support for unnamed tempoaries
; in wasm relocations.
; RUN: not --crash llc -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

@foo = internal global i8* blockaddress(@bar, %addr), align 4

define hidden i32 @bar() #0 {
entry:
  br label %addr

addr:
  ret i32 0
}

; CHECK: LLVM ERROR: relocations for function or section offsets are only supported in metadata sections
