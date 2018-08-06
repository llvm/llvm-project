; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Function with __attribute__((visibility("default")))
define void @defaultVis() #0 {
entry:
  ret void
}

; Function with __attribute__((visibility("hidden")))
define hidden void @hiddenVis() #0 {
entry:
  ret void
}

; CHECK:          SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            defaultVis
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            hiddenVis
; CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:         Function:        1
; CHECK-NEXT: ...
