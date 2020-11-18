; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Import a function just so we can check the index arithmetic for
; WASM_COMDAT_FUNCTION entries is performed correctly
declare i32 @funcImport()
define i32 @callImport() {
entry:
  %call = call i32 @funcImport()
  ret i32 %call
}

; Function in its own COMDAT
$basicInlineFn = comdat any
define linkonce_odr i32 @basicInlineFn() #1 comdat {
  ret i32 0
}

; Global, data, and function in same COMDAT
$sharedComdat = comdat any
@constantData = weak_odr constant [3 x i8] c"abc", comdat($sharedComdat)
define linkonce_odr i32 @sharedFn() #1 comdat($sharedComdat) {
  ret i32 0
}

; CHECK:      Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:      []
; CHECK-NEXT:         ReturnTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:   - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; CHECK-NEXT:           Initial:         0x00000001
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:
; CHECK-NEXT:           Index:           0
; CHECK-NEXT:           ElemType:        FUNCREF
; CHECK-NEXT:           Limits:
; CHECK-NEXT:             Initial:         0x00000000
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           funcImport
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 0, 0 ]
; CHECK-NEXT:  - Type:            DATACOUNT
; CHECK-NEXT:    Count:           1
; CHECK-NEXT:  - Type:            CODE
; CHECK-NEXT:    Relocations:
; CHECK-NEXT:      - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:        Index:           1
; CHECK-NEXT:        Offset:          0x00000004
; CHECK-NEXT:    Functions:
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Locals:
; CHECK-NEXT:        Body:            1080808080000B
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Locals:
; CHECK-NEXT:        Body:            41000B
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Locals:
; CHECK-NEXT:        Body:            41000B
; CHECK-NEXT:  - Type:            DATA
; CHECK-NEXT:    Segments:
; CHECK-NEXT:      - SectionOffset:   6
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           0
; CHECK-NEXT:        Content:         '616263'
; CHECK-NEXT:  - Type:            CUSTOM
; CHECK-NEXT:    Name:            linking
; CHECK-NEXT:    Version:         2
; CHECK-NEXT:    SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            callImport
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Function:        1
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            funcImport
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            basicInlineFn
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Function:        2
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            sharedFn
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Function:        3
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            constantData
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            3
; CHECK-NEXT:    SegmentInfo:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            .rodata.constantData
; CHECK-NEXT:        Alignment:       0
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:    Comdats:
; CHECK-NEXT:        Name:            basicInlineFn
; CHECK-NEXT:        Entries:
; CHECK-NEXT:          - Kind:            FUNCTION
; CHECK-NEXT:            Index:           2
; CHECK-NEXT:        Name:            sharedComdat
; CHECK-NEXT:        Entries:
; CHECK-NEXT:          - Kind:            FUNCTION
; CHECK-NEXT:            Index:           3
; CHECK-NEXT:          - Kind:            DATA
; CHECK-NEXT:            Index:           0
; CHECK-NEXT: ...
