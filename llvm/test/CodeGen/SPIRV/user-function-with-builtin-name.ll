; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that calling a function, that looks like a builtin, but has no matching
; number of arguments is treated as user function.

; "Barrier" with zero args - consider as user function.
; CHECK: OpFunctionCall %[[#]] %[[#]]
define spir_func void @test_too_few() {
  call spir_func void @barrier()
  ret void
}

; Too many args - consider as user function.
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
define spir_func void @test_too_many() {
  call spir_func void @_Z7barrieriiii(i32 1, i32 2, i32 3, i32 4)
  ret void
}

declare spir_func void @barrier()
declare spir_func void @_Z7barrieriiii(i32, i32, i32, i32)
