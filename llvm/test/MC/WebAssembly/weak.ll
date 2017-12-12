; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

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
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           weak_external_data
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   false


; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            weak_function
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:     SymbolInfo:      
; CHECK-NEXT:       - Name:            weak_external_data
; CHECK-NEXT:         Flags:           1
; CHECK-NEXT:       - Name:            weak_function
; CHECK-NEXT:         Flags:           5
; CHECK-NEXT: ...
