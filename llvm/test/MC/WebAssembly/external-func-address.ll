; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s
; Verify that addresses of external functions generate correctly typed
; imports and relocations or type R_TABLE_INDEX_I32.

declare void @f1() #1
@ptr_to_f1 = hidden global void ()* @f1, align 4


; CHECK:   - Type:            IMPORT
; CHECK:     Imports:
; CHECK:       - Module:          env
; CHECK:         Field:           f1
; CHECK:         Kind:            FUNCTION
; CHECK:         SigIndex:        0
; CHECK:   - Type:            ELEM
; CHECK:     Segments:
; CHECK:       - Offset:
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           0
; CHECK:         Functions:       [ 0 ]
; CHECK:   - Type:            DATA
; CHECK:     Relocations:
; CHECK:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_I32
; CHECK:         Index:           0
; CHECK:         Offset:          0x00000006
