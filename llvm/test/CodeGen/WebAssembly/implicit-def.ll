; RUN: llc -o - %s -asm-verbose=false | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test that stackified IMPLICIT_DEF instructions are converted into
; CONST_I32 to provide an explicit push.

; CHECK:      br_if 2,
; CHECK:      i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[L0]]{{$}}
define i1 @f() {
  %a = xor i1 0, 0
  switch i1 %a, label %C [
    i1 0, label %A
    i1 1, label %B
  ]

A:
  %b = xor i1 0, 0
  br label %X

B:
  %c = xor i1 0, 0
  br i1 %c, label %D, label %X

C:
  %d = icmp slt i32 0, 0
  br i1 %d, label %G, label %F

D:
  %e = xor i1 0, 0
  br i1 %e, label %E, label %X

E:
  %f = xor i1 0, 0
  br label %X

F:
  %g = xor i1 0, 0
  br label %G

G:
  %h = phi i1 [ undef, %C ], [ false, %F ]
  br label %X

X:
  %i = phi i1 [ true, %A ], [ true, %B ], [ true, %D ], [ true, %E ], [ %h, %G ]
  ret i1 %i
}

