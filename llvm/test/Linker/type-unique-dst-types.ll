; RUN: llvm-link %p/type-unique-dst-types.ll \
; RUN:           %p/Inputs/type-unique-dst-types2.ll \
; RUN:           %p/Inputs/type-unique-dst-types3.ll -S -o %t1.ll
; RUN: cat %t1.ll | FileCheck %s
; RUN: cat %t1.ll | FileCheck --check-prefix=RENAMED %s

; This tests the importance of keeping track of which types are part of the
; destination module.
; When the second input is merged in, the context gets an unused A.11. When
; the third module is then merged, we should pretend it doesn't exist.

; CHECK: %A = type { %B }
; CHECK-NEXT: %B = type { i8 }

; CHECK: @g3 = external global %A
; CHECK: @g1 = external global %A
; CHECK: @g2 = external global %A

; RENAMED-NOT: A.11

%A = type { %B }
%B = type { i8 }
@g3 = external global %A

define ptr @use_g3() {
  ret ptr @g3
}
