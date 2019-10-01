; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK:        - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            41002802FFFFFFFF0F0B
define i32 @load_i32_from_negative_address() {
  %t = load i32, i32* inttoptr (i32 -1 to i32*)
  ret i32 %t
}

; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            41002802030B
define i32 @load_i32_from_wrapped_address() {
  %t = load i32, i32* inttoptr (i32 4294967299 to i32*)
  ret i32 %t
}
