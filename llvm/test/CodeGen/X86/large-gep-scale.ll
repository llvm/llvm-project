; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; PR5281

; After scaling, this type doesn't fit in memory. Codegen should generate
; correct addressing still.

; The scale is 2147483647*4, which is -4 mod 2^32, so this is u - 4*t and
; folds into a negated scaled index.
; CHECK:      negl %edx
; CHECK-NEXT: leal (%ecx,%edx,4), %eax

define fastcc ptr @_ada_smkr(ptr %u, i32 %t) nounwind {
  %x = getelementptr [2147483647 x i32], ptr %u, i32 %t, i32 0
  ret ptr %x
}
