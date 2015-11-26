; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test the register stackifier pass.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; No because of pointer aliasing.

; CHECK-LABEL: no0:
; CHECK: return $1{{$}}
define i32 @no0(i32* %p, i32* %q) {
  %t = load i32, i32* %q
  store i32 0, i32* %p
  ret i32 %t
}

; No because of side effects.

; CHECK-LABEL: no1:
; CHECK: return $1{{$}}
define i32 @no1(i32* %p, i32* dereferenceable(4) %q) {
  %t = load volatile i32, i32* %q, !invariant.load !0
  store volatile i32 0, i32* %p
  ret i32 %t
}

; Yes because of invariant load and no side effects.

; CHECK-LABEL: yes0:
; CHECK: return $pop0{{$}}
define i32 @yes0(i32* %p, i32* dereferenceable(4) %q) {
  %t = load i32, i32* %q, !invariant.load !0
  store i32 0, i32* %p
  ret i32 %t
}

; Yes because of no intervening side effects.

; CHECK-LABEL: yes1:
; CHECK: return $pop0{{$}}
define i32 @yes1(i32* %q) {
  %t = load volatile i32, i32* %q
  ret i32 %t
}

!0 = !{}
