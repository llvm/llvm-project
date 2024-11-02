; RUN: opt -S -passes=mergefunc < %s | FileCheck %s -implicit-check-not=funC

; Replacments should be totally ordered on the function name.
; If we don't do this we  can end up with one module defining a thunk for @funA
; and another module defining a thunk for @funB.
;
; The problem with this is that the linker could then choose these two stubs
; each of the two modules and we end up with two stubs calling each other.

; CHECK-LABEL: define linkonce_odr i32 @funA
; CHECK-NEXT:    add
; CHECK:         ret

; CHECK-LABEL: define linkonce_odr i32 @funB
; CHECK-NEXT:    tail call i32 @funA(i32 %0, i32 %1)
; CHECK-NEXT:    ret

define linkonce_odr i32 @funC(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum2
  ret i32 %sum3
}

define linkonce_odr i32 @funB(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum2
  ret i32 %sum3
}

define linkonce_odr i32 @funA(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum2
  ret i32 %sum3
}

; This creates a use of @funB, preventing -passes=mergefunc from deleting it.
; @funC, however, can safely be deleted as it has no uses, and is discardable
; if unused.
@take_addr_of_funB = global i8* bitcast (i32 (i32, i32)* @funB to i8*)
