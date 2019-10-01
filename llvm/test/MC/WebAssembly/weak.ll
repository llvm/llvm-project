; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Weak external data reference
@weak_external_data = extern_weak global i32, align 4

; Weak function definition
define weak hidden i32 @weak_function() local_unnamed_addr #0 {
entry:
    %0 = load i32, i32* @weak_external_data, align 4
    ret i32 %0
}

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table


; CHECK:          SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            weak_function
; CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            weak_external_data
; CHECK-NEXT:         Flags:           [ BINDING_WEAK, UNDEFINED ]
; CHECK-NEXT: ...
