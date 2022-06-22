; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; PR5281

; After scaling, this type doesn't fit in memory. Codegen should generate
; correct addressing still.

; CHECK: shll $2, %edx

define fastcc ptr @_ada_smkr(ptr %u, i32 %t) nounwind {
  %x = getelementptr [2147483647 x i32], ptr %u, i32 %t, i32 0
  ret ptr %x
}
