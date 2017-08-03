; RUN: llc < %s -march=x86 | FileCheck %s
; PR5281

; After scaling, this type doesn't fit in memory. Codegen should generate
; correct addressing still.

; CHECK: shll $2, %edx

define fastcc i32* @_ada_smkr([2147483647 x i32]* %u, i32 %t) nounwind {
  %x = getelementptr [2147483647 x i32], [2147483647 x i32]* %u, i32 %t, i32 0
  ret i32* %x
}
