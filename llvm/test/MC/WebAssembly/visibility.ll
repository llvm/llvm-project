; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

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

; CHECK:        - Type:            CUSTOM

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:     SymbolInfo:
; CHECK-NEXT:       - Name:            hiddenVis
; CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT: ...
