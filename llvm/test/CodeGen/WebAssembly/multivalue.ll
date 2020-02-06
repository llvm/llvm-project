; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+multivalue,+tail-call | FileCheck %s
; RUN: llc < %s --filetype=obj -mattr=+multivalue,+tail-call | obj2yaml | FileCheck %s --check-prefix OBJ

; Test that the multivalue calls, returns, function types, and block
; types work as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%pair = type { i32, i64 }
%packed_pair = type <{ i32, i64 }>


; CHECK-LABEL: pair_const:
; CHECK-NEXT: .functype pair_const () -> (i32, i64)
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 42{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 42{{$}}
; CHECK-NEXT: return $pop[[L0]], $pop[[L1]]{{$}}
define %pair @pair_const() {
  ret %pair { i32 42, i64 42 }
}

; CHECK-LABEL: packed_pair_const:
; CHECK-NEXT: .functype packed_pair_const () -> (i32, i64)
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 42{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 42{{$}}
; CHECK-NEXT: return $pop[[L0]], $pop[[L1]]{{$}}
define %packed_pair @packed_pair_const() {
  ret %packed_pair <{ i32 42, i64 42 }>
}

; CHECK-LABEL: pair_ident:
; CHECK-NEXT: .functype pair_ident (i32, i64) -> (i32, i64)
; CHECK-NEXT: return $0, $1{{$}}
define %pair @pair_ident(%pair %p) {
  ret %pair %p
}

; CHECK-LABEL: packed_pair_ident:
; CHECK-NEXT: .functype packed_pair_ident (i32, i64) -> (i32, i64)
; CHECK-NEXT: return $0, $1{{$}}
define %packed_pair @packed_pair_ident(%packed_pair %p) {
  ret %packed_pair %p
}

;; TODO: Multivalue calls are a WIP and do not necessarily produce
;; correct output. For now, just check that they don't cause any
;; crashes.

define void @pair_call() {
  %p = call %pair @pair_const()
  ret void
}

define void @packed_pair_call() {
  %p = call %packed_pair @packed_pair_const()
  ret void
}

define %pair @pair_call_return() {
  %p = call %pair @pair_const()
  ret %pair %p
}

define %packed_pair @packed_pair_call_return() {
  %p = call %packed_pair @packed_pair_const()
  ret %packed_pair %p
}

define %pair @pair_tail_call() {
  %p = musttail call %pair @pair_const()
  ret %pair %p
}

define %packed_pair @packed_pair_tail_call() {
  %p = musttail call %packed_pair @packed_pair_const()
  ret %packed_pair %p
}

define i32 @pair_call_return_first() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 0
  ret i32 %v
}

define i32 @packed_pair_call_return_first() {
  %p = call %packed_pair @packed_pair_const()
  %v = extractvalue %packed_pair %p, 0
  ret i32 %v
}

define i64 @pair_call_return_second() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 1
  ret i64 %v
}

define i64 @packed_pair_call_return_second() {
  %p = call %packed_pair @packed_pair_const()
  %v = extractvalue %packed_pair %p, 1
  ret i64 %v
}

define %pair @pair_pass_through(%pair %p) {
  %r = call %pair @pair_ident(%pair %p)
  ret %pair %r
}

define %packed_pair @packed_pair_pass_through(%packed_pair %p) {
  %r = call %packed_pair @packed_pair_ident(%packed_pair %p)
  ret %packed_pair %r
}

; CHECK-LABEL: minimal_loop:
; CHECK-NEXT: .functype minimal_loop (i32) -> (i32, i64)
; CHECK-NEXT: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: loop () -> (i32, i64)
; CHECK-NEXT: br 0{{$}}
; CHECK-NEXT: .LBB{{[0-9]+}}_2:
; CHECK-NEXT: end_loop{{$}}
define %pair @minimal_loop(i32* %p) {
entry:
  br label %loop
loop:
  br label %loop
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 2
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 10
; CHECK-NEXT: .ascii "multivalue"
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 9
; CHECK-NEXT: .ascii "tail-call"

; OBJ-LABEL:  - Type:            TYPE
; OBJ-NEXT:     Signatures:
; OBJ-NEXT:       - Index:           0
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
; OBJ-NEXT:       - Index:           1
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
; OBJ-NEXT:       - Index:           2
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:     []
; OBJ-NEXT:       - Index:           3
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:       - Index:           4
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I64
; OBJ-NEXT:       - Index:           5
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
