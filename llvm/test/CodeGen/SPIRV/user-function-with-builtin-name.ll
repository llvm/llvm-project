; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; Verify that calling a function, that looks like a builtin, but has no matching
; number of arguments is treated as user function.

; CHECK: OpFunctionCall %[[#]] %[[#]]
define spir_func void @test() {
  call spir_func void @barrier()
  ret void
}

declare spir_func void @barrier()
