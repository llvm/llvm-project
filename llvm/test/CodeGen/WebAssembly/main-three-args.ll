; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that main function with a non-standard third argument is
; not wrapped.

target triple = "wasm32-unknown-unknown"

define i32 @main(i32 %a, ptr %b, ptr %c) {
  ret i32 0
}

; CHECK-LABEL: main:
; CHECK-NEXT: .functype main (i32, i32, i32) -> (i32)

; CHECK-NOT: __original_main:
