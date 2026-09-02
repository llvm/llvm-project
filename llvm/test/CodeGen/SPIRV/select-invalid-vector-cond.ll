; RUN: not --crash llc -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; __spirv_Select can pair a vector boolean condition with scalar operands,
; which is malformed for OpSelect and must be diagnosed.

; CHECK: LLVM ERROR: OpSelect with a scalar result requires a scalar boolean condition

define spir_kernel void @bad_select(i32 %a, i32 %b, ptr addrspace(1) %out, <4 x i1> %cond) {
entry:
  %call = call spir_func i32 @_Z14__spirv_SelectDv4_bii(<4 x i1> %cond, i32 %a, i32 %b)
  store i32 %call, ptr addrspace(1) %out
  ret void
}

declare spir_func i32 @_Z14__spirv_SelectDv4_bii(<4 x i1>, i32, i32)
