; RUN: llvm-link -S %s %p/Inputs/type-unique-opaque.ll | FileCheck %s

; Test that a failed attempt at merging %u2 and %t2 (for the other file) will
; not cause %u and %t to get merged.

; CHECK: %u = type opaque
; CHECK: define %u @g(%u %a) {

%u = type opaque
%u2 = type { %u, i8 }

declare %u2 @f()

define %u @g(%u %a) {
  ret %u %a
}
