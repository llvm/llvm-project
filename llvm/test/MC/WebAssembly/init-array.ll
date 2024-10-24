; RUN: llc -mcpu=mvp -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

@p_init1 = hidden global ptr @init1, section ".init_array", align 4
@p_init2 = hidden global ptr @init2, section ".init_array", align 4

define hidden void @init1() #0 { ret void }
define hidden void @init2() #0 { ret void }


; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; CHECK-NEXT:           Minimum:         0x0
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:
; CHECK-NEXT:           Index:           0
; CHECK-NEXT:           ElemType:        FUNCREF
; CHECK-NEXT:           Limits:
; CHECK-NEXT:             Minimum:         0x0
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 0 ]
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            0B
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            0B
; CHECK-NEXT:    - Type:            CUSTOM
; CHECK-NEXT:      Name:            linking
; CHECK-NEXT:      Version:         2
; CHECK-NEXT:      SymbolTable:
; CHECK-NEXT:        - Index:           0
; CHECK-NEXT:          Kind:            FUNCTION
; CHECK-NEXT:          Name:            init1
; CHECK-NEXT:          Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:          Function:        0
; CHECK-NEXT:        - Index:           1
; CHECK-NEXT:          Kind:            FUNCTION
; CHECK-NEXT:          Name:            init2
; CHECK-NEXT:          Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:          Function:        1
; CHECK-NEXT:      InitFunctions:
; CHECK-NEXT:        - Priority:        65535
; CHECK-NEXT:          Symbol:          0
; CHECK-NEXT:        - Priority:        65535
; CHECK-NEXT:          Symbol:          1
; CHECK-NEXT: ...
