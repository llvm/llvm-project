; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+tail-call | FileCheck --check-prefixes=CHECK,SLOW %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel -mattr=+tail-call | FileCheck --check-prefixes=CHECK,FAST %s
; RUN: llc < %s --filetype=obj -mattr=+tail-call | obj2yaml | FileCheck --check-prefix=YAML %s

; Test that the tail calls lower correctly

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%fn = type <{i32 (%fn, i32, i32)*}>
declare i1 @foo(i1)
declare i1 @bar(i1)

; CHECK-LABEL: recursive_notail_nullary:
; CHECK: {{^}} call recursive_notail_nullary{{$}}
; CHECK-NEXT: return
define void @recursive_notail_nullary() {
  notail call void @recursive_notail_nullary()
  ret void
}

; CHECK-LABEL: recursive_musttail_nullary:
; CHECK: return_call recursive_musttail_nullary{{$}}
define void @recursive_musttail_nullary() {
  musttail call void @recursive_musttail_nullary()
  ret void
}

; CHECK-LABEL: recursive_tail_nullary:
; SLOW: return_call recursive_tail_nullary{{$}}
; FAST: {{^}} call recursive_tail_nullary{{$}}
; FAST-NEXT: return{{$}}
define void @recursive_tail_nullary() {
  tail call void @recursive_tail_nullary()
  ret void
}

; CHECK-LABEL: recursive_notail:
; CHECK: call $push[[L:[0-9]+]]=, recursive_notail, $0, $1{{$}}
; CHECK-NEXT: return $pop[[L]]{{$}}
define i32 @recursive_notail(i32 %x, i32 %y) {
  %v = notail call i32 @recursive_notail(i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: recursive_musttail:
; CHECK: return_call recursive_musttail, $0, $1{{$}}
define i32 @recursive_musttail(i32 %x, i32 %y) {
  %v = musttail call i32 @recursive_musttail(i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: recursive_tail:
; SLOW: return_call recursive_tail, $0, $1{{$}}
; FAST: call $push[[L:[0-9]+]]=, recursive_tail, $0, $1{{$}}
; FAST-NEXT: return $pop[[L]]{{$}}
define i32 @recursive_tail(i32 %x, i32 %y) {
  %v = tail call i32 @recursive_tail(i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: indirect_notail:
; CHECK: call_indirect $push[[L:[0-9]+]]=, $0, $1, $2, $0{{$}}
; CHECK-NEXT: return $pop[[L]]{{$}}
define i32 @indirect_notail(%fn %f, i32 %x, i32 %y) {
  %p = extractvalue %fn %f, 0
  %v = notail call i32 %p(%fn %f, i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: indirect_musttail:
; CHECK: return_call_indirect , $0, $1, $2, $0{{$}}
define i32 @indirect_musttail(%fn %f, i32 %x, i32 %y) {
  %p = extractvalue %fn %f, 0
  %v = musttail call i32 %p(%fn %f, i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: indirect_tail:
; CHECK: return_call_indirect , $0, $1, $2, $0{{$}}
define i32 @indirect_tail(%fn %f, i32 %x, i32 %y) {
  %p = extractvalue %fn %f, 0
  %v = tail call i32 %p(%fn %f, i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: choice_notail:
; CHECK: call_indirect $push[[L:[0-9]+]]=, $0, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[L]]{{$}}
define i1 @choice_notail(i1 %x) {
  %p = select i1 %x, i1 (i1)* @foo, i1 (i1)* @bar
  %v = notail call i1 %p(i1 %x)
  ret i1 %v
}

; CHECK-LABEL: choice_musttail:
; CHECK: return_call_indirect , $0, $pop{{[0-9]+}}{{$}}
define i1 @choice_musttail(i1 %x) {
  %p = select i1 %x, i1 (i1)* @foo, i1 (i1)* @bar
  %v = musttail call i1 %p(i1 %x)
  ret i1 %v
}

; CHECK-LABEL: choice_tail:
; SLOW: return_call_indirect , $0, $pop{{[0-9]+}}{{$}}
; FAST: call_indirect $push[[L:[0-9]+]]=, $0, $pop{{[0-9]+}}{{$}}
; FAST: return $pop[[L]]{{$}}
define i1 @choice_tail(i1 %x) {
  %p = select i1 %x, i1 (i1)* @foo, i1 (i1)* @bar
  %v = tail call i1 %p(i1 %x)
  ret i1 %v
}

; It is an LLVM validation error for a 'musttail' callee to have a different
; prototype than its caller, so the following tests can only be done with
; 'tail'.

; CHECK-LABEL: mismatched_prototypes:
; SLOW: return_call baz, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; FAST: call $push[[L:[0-9]+]]=, baz, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; FAST: return $pop[[L]]{{$}}
declare i32 @baz(i32, i32, i32)
define i32 @mismatched_prototypes() {
  %v = tail call i32 @baz(i32 0, i32 42, i32 6)
  ret i32 %v
}

; CHECK-LABEL: mismatched_return_void:
; CHECK: call $drop=, baz, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: return{{$}}
define void @mismatched_return_void() {
  %v = tail call i32 @baz(i32 0, i32 42, i32 6)
  ret void
}

; CHECK-LABEL: mismatched_return_f32:
; CHECK: call $push[[L:[0-9]+]]=, baz, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: f32.reinterpret_i32 $push[[L1:[0-9]+]]=, $pop[[L]]{{$}}
; CHECK: return $pop[[L1]]{{$}}
define float @mismatched_return_f32() {
  %v = tail call i32 @baz(i32 0, i32 42, i32 6)
  %u = bitcast i32 %v to float
  ret float %u
}

; CHECK-LABEL: mismatched_indirect_void:
; CHECK: call_indirect $drop=, $0, $1, $2, $0{{$}}
; CHECK: return{{$}}
define void @mismatched_indirect_void(%fn %f, i32 %x, i32 %y) {
  %p = extractvalue %fn %f, 0
  %v = tail call i32 %p(%fn %f, i32 %x, i32 %y)
  ret void
}

; CHECK-LABEL: mismatched_indirect_f32:
; CHECK: call_indirect $push[[L:[0-9]+]]=, $0, $1, $2, $0{{$}}
; CHECK: f32.reinterpret_i32 $push[[L1:[0-9]+]]=, $pop[[L]]{{$}}
; CHECK: return $pop[[L1]]{{$}}
define float @mismatched_indirect_f32(%fn %f, i32 %x, i32 %y) {
  %p = extractvalue %fn %f, 0
  %v = tail call i32 %p(%fn %f, i32 %x, i32 %y)
  %u = bitcast i32 %v to float
  ret float %u
}

; CHECK-LABEL: mismatched_byval:
; CHECK: i32.store
; CHECK: return_call quux, $pop{{[0-9]+}}{{$}}
declare i32 @quux(i32* byval(i32))
define i32 @mismatched_byval(i32* %x) {
  %v = tail call i32 @quux(i32* byval(i32) %x)
  ret i32 %v
}

; CHECK-LABEL: varargs:
; CHECK: i32.store
; CHECK: call $0=, var, $1{{$}}
; CHECK: return $0{{$}}
declare i32 @var(...)
define i32 @varargs(i32 %x) {
  %v = tail call i32 (...) @var(i32 %x)
  ret i32 %v
}

; Type transformations inhibit tail calls, even when they are nops

; CHECK-LABEL: mismatched_return_zext:
; CHECK: call
define i32 @mismatched_return_zext() {
  %v = tail call i1 @foo(i1 1)
  %u = zext i1 %v to i32
  ret i32 %u
}

; CHECK-LABEL: mismatched_return_sext:
; CHECK: call
define i32 @mismatched_return_sext() {
  %v = tail call i1 @foo(i1 1)
  %u = sext i1 %v to i32
  ret i32 %u
}

; CHECK-LABEL: mismatched_return_trunc:
; CHECK: call
declare i32 @int()
define i1 @mismatched_return_trunc() {
  %v = tail call i32 @int()
  %u = trunc i32 %v to i1
  ret i1 %u
}

; Stack-allocated arguments inhibit tail calls

; CHECK-LABEL: stack_arg:
; CHECK: call
define i32 @stack_arg(i32* %x) {
  %a = alloca i32
  %v = tail call i32 @stack_arg(i32* %a)
  ret i32 %v
}

; CHECK-LABEL: stack_arg_gep:
; CHECK: call
define i32 @stack_arg_gep(i32* %x) {
  %a = alloca { i32, i32 }
  %p = getelementptr { i32, i32 }, { i32, i32 }* %a, i32 0, i32 1
  %v = tail call i32 @stack_arg_gep(i32* %p)
  ret i32 %v
}

; CHECK-LABEL: stack_arg_cast:
; CHECK: global.get $push{{[0-9]+}}=, __stack_pointer
; CHECK: global.set __stack_pointer, $pop{{[0-9]+}}
; FAST: call ${{[0-9]+}}=, stack_arg_cast, $pop{{[0-9]+}}
; CHECK: global.set __stack_pointer, $pop{{[0-9]+}}
; SLOW: return_call stack_arg_cast, ${{[0-9]+}}
define i32 @stack_arg_cast(i32 %x) {
  %a = alloca [64 x i32]
  %i = ptrtoint [64 x i32]* %a to i32
  %v = tail call i32 @stack_arg_cast(i32 %i)
  ret i32 %v
}

; Check that the signatures generated for external indirectly
; return-called functions include the proper return types

; YAML-LABEL: - Index:           8
; YAML-NEXT:    ParamTypes:
; YAML-NEXT:      - I32
; YAML-NEXT:      - F32
; YAML-NEXT:      - I64
; YAML-NEXT:      - F64
; YAML-NEXT:    ReturnTypes:
; YAML-NEXT:      - I32
define i32 @unique_caller(i32 (i32, float, i64, double)** %p) {
  %f = load i32 (i32, float, i64, double)*, i32 (i32, float, i64, double)** %p
  %v = tail call i32 %f(i32 0, float 0., i64 0, double 0.)
  ret i32 %v
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 9
; CHECK-NEXT: .ascii "tail-call"
