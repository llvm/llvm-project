; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+tail-call | FileCheck --check-prefixes=CHECK,SLOW %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel -mattr=+tail-call | FileCheck --check-prefixes=CHECK,FAST %s

; Test that the tail-call attribute is accepted

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
; CHECK: i32.call $push[[L:[0-9]+]]=, recursive_notail, $0, $1{{$}}
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
; FAST: i32.call $push[[L:[0-9]+]]=, recursive_tail, $0, $1{{$}}
; FAST-NEXT: return $pop[[L]]{{$}}
define i32 @recursive_tail(i32 %x, i32 %y) {
  %v = tail call i32 @recursive_tail(i32 %x, i32 %y)
  ret i32 %v
}

; CHECK-LABEL: indirect_notail:
; CHECK: i32.call_indirect $push[[L:[0-9]+]]=, $0, $1, $2, $0{{$}}
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
; CHECK: i32.call_indirect $push[[L:[0-9]+]]=, $0, $pop{{[0-9]+}}{{$}}
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
; FAST: i32.call_indirect $push[[L:[0-9]+]]=, $0, $pop{{[0-9]+}}{{$}}
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
; FAST: i32.call $push[[L:[0-9]+]]=, baz, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; FAST: return $pop[[L]]{{$}}
declare i32 @baz(i32, i32, i32)
define i32 @mismatched_prototypes() {
  %v = tail call i32 @baz(i32 0, i32 42, i32 6)
  ret i32 %v
}

; CHECK-LABEL: mismatched_byval:
; CHECK: i32.store
; CHECK: return_call quux, $pop{{[0-9]+}}{{$}}
declare i32 @quux(i32* byval)
define i32 @mismatched_byval(i32* %x) {
  %v = tail call i32 @quux(i32* byval %x)
  ret i32 %v
}

; CHECK-LABEL: varargs:
; CHECK: i32.store
; CHECK: return_call var, $1{{$}}
declare i32 @var(...)
define i32 @varargs(i32 %x) {
  %v = tail call i32 (...) @var(i32 %x)
  ret i32 %v
}

; CHECK-LABEL: .section .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 9
; CHECK-NEXT: .ascii "tail-call"
