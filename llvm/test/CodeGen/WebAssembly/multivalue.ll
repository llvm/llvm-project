; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+multivalue | FileCheck %s
; RUN: llc < %s --filetype=obj -mattr=+multivalue | obj2yaml | FileCheck %s --check-prefix OBJ

; Test that the multivalue returns, function types, and block types
; work as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%pair = type { i32, i32 }
%packed_pair = type <{ i32, i32 }>

; CHECK-LABEL: pair_ident:
; CHECK-NEXT: .functype pair_ident (i32, i32) -> (i32, i32)
; CHECK-NEXT: return $0, $1{{$}}
define %pair @pair_ident(%pair %p) {
  ret %pair %p
}

; CHECK-LABEL: packed_pair_ident:
; CHECK-NEXT: .functype packed_pair_ident (i32, i32) -> (i32, i32)
; CHECK-NEXT: return $0, $1{{$}}
define %packed_pair @packed_pair_ident(%packed_pair %p) {
  ret %packed_pair %p
}

; CHECK-LABEL: minimal_loop:
; CHECK-NEXT: .functype minimal_loop (i32) -> (i32, i64)
; CHECK-NEXT: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: loop () -> (i32, i64)
; CHECK-NEXT: br 0{{$}}
; CHECK-NEXT: .LBB{{[0-9]+}}_2:
; CHECK-NEXT: end_loop{{$}}
define {i32, i64} @minimal_loop(i32* %p) {
entry:
  br label %loop
loop:
  br label %loop
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 10
; CHECK-NEXT: .ascii "multivalue"

; OBJ-LABEL:  - Type:            TYPE
; OBJ-NEXT:     Signatures:
; OBJ-NEXT:       - Index:           0
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I32
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I32
; OBJ-NEXT:       - Index:           1
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
