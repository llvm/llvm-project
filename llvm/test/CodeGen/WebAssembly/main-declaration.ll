; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -opaque-pointers | FileCheck %s

; Test main functions with alternate signatures.

target triple = "wasm32-unknown-unknown"

declare i32 @main()

define i32 @foo() {
  %t = call i32 @main()
  ret i32 %t
}

; CHECK-LABEL: foo:
; CHECK-NEXT:    .functype foo () -> (i32)
; CHECK-NEXT:    call __original_main
; CHECK-NEXT:    end_function
