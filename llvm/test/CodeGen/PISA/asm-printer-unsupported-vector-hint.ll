; RUN: not llc -mtriple=pisa -filetype=asm %s -o /dev/null 2>&1 | FileCheck %s

target triple = "pisa"

define pisa_kernel void @unsupported_vector_hint() !vec_type_hint !0 {
  ret void
}

!0 = !{<2 x i2> zeroinitializer, i32 0}

; CHECK: LLVM ERROR: unsupported PISA kernel vector type hint integer
