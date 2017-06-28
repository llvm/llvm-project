; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

; foo_alias() function is weak alias of function foo()
; Generates two exports of the same function, one of them weak

@foo_alias = weak hidden alias i32 (...), bitcast (i32 ()* @foo to i32 (...)*)

define hidden i32 @foo() #0 {
entry:
  ret i32 0
}

; CHECK:        - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            foo_alias
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0


; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:     DataAlignment:   0
; CHECK-NEXT:     SymbolInfo:      
; CHECK-NEXT:       - Name:            foo_alias
; CHECK-NEXT:         Flags:           1
; CHECK-NEXT: ...
