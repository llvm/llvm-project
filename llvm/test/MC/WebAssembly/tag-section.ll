; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %s -o - | obj2yaml | FileCheck %s
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %s -o - | llvm-readobj -S - | FileCheck -check-prefix=SEC %s

target triple = "wasm32-unknown-unknown"

declare void @llvm.wasm.throw(i32, ptr)

define i32 @test_throw0(ptr %p) {
  call void @llvm.wasm.throw(i32 0, ptr %p)
  ret i32 0
}

define i32 @test_throw1(ptr %p) {
  call void @llvm.wasm.throw(i32 0, ptr %p)
  ret i32 1
}

; CHECK:      Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:      []

; CHECK:        - Type:            TAG
; CHECK-NEXT:     TagTypes:        [ 1 ]

; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_TAG_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x6
; CHECK-NEXT:       - Type:            R_WASM_TAG_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x11

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:

; CHECK:            - Index:           1
; CHECK-NEXT:         Kind:            TAG
; CHECK-NEXT:         Name:            __cpp_exception
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Tag:             0

; SEC:          Type: TAG (0xD)
; SEC-NEXT:     Size: 3
; SEC-NEXT:     Offset: 69
