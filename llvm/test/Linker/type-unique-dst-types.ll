; RUN: llvm-link %p/type-unique-dst-types.ll \
; RUN:           %p/Inputs/type-unique-dst-types2.ll \
; RUN:           %p/Inputs/type-unique-dst-types3.ll -S -o %t1.ll
; RUN: cat %t1.ll | FileCheck %s

; The types of @g1 and @g3 can be deduplicated, but @g2 should retain its
; opaque type, even if it has the same name as a type from a different module.

; CHECK: %A = type { %B }
; CHECK-NEXT: %B = type { i8 }
; CHECK-NEXT: %A.11 = type opaque

; CHECK: @g3 = external global %A
; CHECK: @g1 = external global %A
; CHECK: @g2 = external global %A.11

%A = type { %B }
%B = type { i8 }
@g3 = external global %A

define ptr @use_g3() {
  ret ptr @g3
}
