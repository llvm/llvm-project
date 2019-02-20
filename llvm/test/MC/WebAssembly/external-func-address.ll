; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Verify that addresses of external functions generate correctly typed
; imports and relocations of type R_TABLE_INDEX_I32.

declare void @f0(i32) #0
@ptr_to_f0 = hidden global void (i32)* @f0, align 4

attributes #0 = { "wasm-import-module"="somewhere" }

declare void @f1(i32) #1
@ptr_to_f1 = hidden global void (i32)* @f1, align 4

; Check that varargs functions have correctly typed imports
declare i32 @varargs(i32, i32, ...)
define void @call(i32) {
 %a = call i32 (i32, i32, ...) @varargs(i32 0, i32 0)
 ret void
}

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:      
; CHECK-NEXT:   Version:         0x00000001
; CHECK-NEXT: Sections:        
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:      
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           varargs
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK:            - Module:          somewhere
; CHECK-NEXT:         Field:           f0
; CHECK:            - Module:          env
; CHECK-NEXT:         Field:           f1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK:        - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 1, 2 ]
; CHECK:        - Type:            DATA
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_I32
; CHECK-NEXT:         Index:           3
; CHECK-NEXT:         Offset:          0x00000006
