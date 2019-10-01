; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown -o %t.o %s
; RUN: llvm-nm --extern-only %t.o | FileCheck %s

; Verity that hidden symbols are listed even when --extern-only is passed

define hidden i32 @foo() {
entry:
  ret i32 42
}

define i32 @bar() {
entry:
  ret i32 43
}

define internal i32 @baz() {
entry:
  ret i32 44
}

; CHECK: 00000006 T bar
; CHECK-NOT: baz
; CHECK: 00000001 T foo
