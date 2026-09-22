; RUN: not llc -mtriple=pisa -filetype=asm %s -o /dev/null 2>&1 | FileCheck %s

target triple = "pisa"

define void @k_fp128() !vec_type_hint !0 {
  ret void
}

!0 = !{fp128 poison, i32 0}

; CHECK: LLVM ERROR: unsupported PISA kernel vector type hint{{$}}
