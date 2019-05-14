; RUN: llc -filetype=obj -exception-model=wasm -mattr=+exception-handling %s -o - | obj2yaml | FileCheck %s
; RUN: llc -filetype=obj -exception-model=wasm -mattr=+exception-handling %s -o - | llvm-readobj -S | FileCheck -check-prefix=SEC %s

target triple = "wasm32-unknown-unknown"

declare void @llvm.wasm.throw(i32, i8*)

define i32 @test_throw0(i8* %p) {
  call void @llvm.wasm.throw(i32 0, i8* %p)
  ret i32 0
}

define i32 @test_throw1(i8* %p) {
  call void @llvm.wasm.throw(i32 0, i8* %p)
  ret i32 1
}

; CHECK:      Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32

; CHECK:        - Type:            EVENT
; CHECK-NEXT:     Events:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Attribute:       0
; CHECK-NEXT:         SigIndex:        1

; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_EVENT_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000006
; CHECK-NEXT:       - Type:            R_WASM_EVENT_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000011

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:

; CHECK:            - Index:           1
; CHECK-NEXT:         Kind:            EVENT
; CHECK-NEXT:         Name:            __cpp_exception
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Event:           0

; SEC:          Type: EVENT (0xD)
; SEC-NEXT:     Size: 3
; SEC-NEXT:     Offset: 97
