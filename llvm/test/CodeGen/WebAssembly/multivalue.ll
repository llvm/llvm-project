; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -mcpu=mvp -mattr=+multivalue,+tail-call | FileCheck %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -mcpu=mvp -mattr=+reference-types,+multivalue,+tail-call | FileCheck --check-prefix REF %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=+multivalue,+tail-call | FileCheck %s --check-prefix REGS
; RUN: llc < %s --filetype=obj -mcpu=mvp -mattr=+multivalue,+tail-call | obj2yaml | FileCheck %s --check-prefix OBJ

; Test that the multivalue calls, returns, function types, and block
; types work as expected.

target triple = "wasm32-unknown-unknown"

%pair = type { i32, i64 }
%rpair = type { i64, i32 }

declare void @use_i32(i32)
declare void @use_i64(i64)

; CHECK-LABEL: pair_const:
; CHECK-NEXT: .functype pair_const () -> (i32, i64)
; CHECK-NEXT: i32.const 42{{$}}
; CHECK-NEXT: i64.const 42{{$}}
; CHECK-NEXT: end_function{{$}}
define %pair @pair_const() {
  ret %pair { i32 42, i64 42 }
}

; CHECK-LABEL: pair_ident:
; CHECK-NEXT: .functype pair_ident (i32, i64) -> (i32, i64)
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: end_function{{$}}
define %pair @pair_ident(%pair %p) {
  ret %pair %p
}

; CHECK-LABEL: pair_call:
; CHECK-NEXT: .functype pair_call () -> ()
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $drop=, $drop=, pair_const{{$}}
define void @pair_call() {
  %p = call %pair @pair_const()
  ret void
}

; CHECK-LABEL: pair_call_return:
; CHECK-NEXT: .functype pair_call_return () -> (i32, i64)
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $push{{[0-9]+}}=, $push{{[0-9]+}}=, pair_const{{$}}
define %pair @pair_call_return() {
  %p = call %pair @pair_const()
  ret %pair %p
}

; CHECK-LABEL: pair_call_indirect:
; CHECK-NEXT: .functype pair_call_indirect (i32) -> (i32, i64)
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: call_indirect () -> (i32, i64){{$}}
; REF:        call_indirect __indirect_function_table, () -> (i32, i64){{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call_indirect $push{{[0-9]+}}=, $push{{[0-9]+}}=, $0{{$}}
define %pair @pair_call_indirect(ptr %f) {
  %p = call %pair %f()
  ret %pair %p
}

; CHECK-LABEL: pair_tail_call:
; CHECK-NEXT: .functype pair_tail_call () -> (i32, i64)
; CHECK-NEXT: return_call pair_const{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: return_call pair_const{{$}}
define %pair @pair_tail_call() {
  %p = musttail call %pair @pair_const()
  ret %pair %p
}

; CHECK-LABEL: pair_call_return_first:
; CHECK-NEXT: .functype pair_call_return_first () -> (i32)
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $push{{[0-9]+}}=, $drop=, pair_const{{$}}
define i32 @pair_call_return_first() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 0
  ret i32 %v
}

; CHECK-LABEL: pair_call_return_second:
; CHECK-NEXT: .functype pair_call_return_second () -> (i64)
; CHECK-NEXT: .local i64{{$}}
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $drop=, $0=, pair_const{{$}}
define i64 @pair_call_return_second() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 1
  ret i64 %v
}

; CHECK-LABEL: pair_call_use_first:
; CHECK-NEXT: .functype pair_call_use_first () -> ()
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: call use_i32{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $push{{[0-9]+}}=, $drop=, pair_const{{$}}
define void @pair_call_use_first() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 0
  call void @use_i32(i32 %v)
  ret void
}

; CHECK-LABEL: pair_call_use_second:
; CHECK-NEXT: .functype pair_call_use_second () -> ()
; CHECK-NEXT: .local i64
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: drop{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: call use_i64{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $drop=, $0=, pair_const{{$}}
define void @pair_call_use_second() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 1
  call void @use_i64(i64 %v)
  ret void
}

; CHECK-LABEL: pair_call_use_first_return_second:
; CHECK-NEXT: .functype pair_call_use_first_return_second () -> (i64)
; CHECK-NEXT: .local i64{{$}}
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: call use_i32{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $push{{[0-9]+}}=, $0=, pair_const{{$}}
define i64 @pair_call_use_first_return_second() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 0
  call void @use_i32(i32 %v)
  %r = extractvalue %pair %p, 1
  ret i64 %r
}

; CHECK-LABEL: pair_call_use_second_return_first:
; CHECK-NEXT: .functype pair_call_use_second_return_first () -> (i32)
; CHECK-NEXT: .local i32, i64{{$}}
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: local.set 1{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: call use_i64{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $0=, $1=, pair_const{{$}}
define i32 @pair_call_use_second_return_first() {
  %p = call %pair @pair_const()
  %v = extractvalue %pair %p, 1
  call void @use_i64(i64 %v)
  %r = extractvalue %pair %p, 0
  ret i32 %r
}

; CHECK-LABEL: pair_pass_through:
; CHECK-NEXT: .functype pair_pass_through (i32, i64) -> (i32, i64)
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: call pair_ident{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $push{{[0-9]+}}=, $push{{[0-9]+}}=, pair_ident, $0, $1{{$}}
define %pair @pair_pass_through(%pair %p) {
  %r = call %pair @pair_ident(%pair %p)
  ret %pair %r
}

; CHECK-LABEL: pair_swap:
; CHECK-NEXT: .functype pair_swap (i32, i64) -> (i64, i32)
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
define %rpair @pair_swap(%pair %p) {
  %first = extractvalue %pair %p, 0
  %second = extractvalue %pair %p, 1
  %r1 = insertvalue %rpair undef, i32 %first, 1
  %r2 = insertvalue %rpair %r1, i64 %second, 0
  ret %rpair %r2
}

; CHECK-LABEL: pair_call_swap:
; CHECK-NEXT: .functype pair_call_swap () -> (i64, i32)
; CHECK-NEXT: .local i32, i64{{$}}
; CHECK-NEXT: call pair_const{{$}}
; CHECK-NEXT: local.set 1{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $0=, $1=, pair_const{{$}}
define %rpair @pair_call_swap() {
  %p = call %pair @pair_const()
  %first = extractvalue %pair %p, 0
  %second = extractvalue %pair %p, 1
  %r1 = insertvalue %rpair undef, i32 %first, 1
  %r2 = insertvalue %rpair %r1, i64 %second, 0
  ret %rpair %r2
}

; CHECK-LABEL: pair_pass_through_swap:
; CHECK-NEXT: .functype pair_pass_through_swap (i32, i64) -> (i64, i32)
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: call pair_ident{{$}}
; CHECK-NEXT: local.set 1{{$}}
; CHECK-NEXT: local.set 0{{$}}
; CHECK-NEXT: local.get 1{{$}}
; CHECK-NEXT: local.get 0{{$}}
; CHECK-NEXT: end_function{{$}}
; REGS: call $0=, $1=, pair_ident, $0, $1{{$}}
define %rpair @pair_pass_through_swap(%pair %p) {
  %p1 = call %pair @pair_ident(%pair %p)
  %first = extractvalue %pair %p1, 0
  %second = extractvalue %pair %p1, 1
  %r1 = insertvalue %rpair undef, i32 %first, 1
  %r2 = insertvalue %rpair %r1, i64 %second, 0
  ret %rpair %r2
}

; CHECK-LABEL: minimal_loop:
; CHECK-NEXT: .functype minimal_loop (i32) -> (i32, i64)
; CHECK-NEXT: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: loop () -> (i32, i64)
; CHECK-NEXT: br 0{{$}}
; CHECK-NEXT: .LBB{{[0-9]+}}_2:
; CHECK-NEXT: end_loop{{$}}
; CHECK-NEXT: end_function{{$}}
define %pair @minimal_loop(ptr %p) {
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
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
; OBJ-NEXT:       - Index:           4
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:       - Index:           5
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I64
; OBJ-NEXT:       - Index:           6
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:         ReturnTypes:     []
; OBJ-NEXT:       - Index:           7
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I64
; OBJ-NEXT:         ReturnTypes:     []
; OBJ-NEXT:       - Index:           8
; OBJ-NEXT:         ParamTypes:
; OBJ-NEXT:           - I32
; OBJ-NEXT:           - I64
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I64
; OBJ-NEXT:           - I32
; OBJ-NEXT:       - Index:           9
; OBJ-NEXT:         ParamTypes:      []
; OBJ-NEXT:         ReturnTypes:
; OBJ-NEXT:           - I64
; OBJ-NEXT:           - I32
