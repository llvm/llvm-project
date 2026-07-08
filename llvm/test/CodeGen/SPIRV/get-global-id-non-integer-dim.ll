; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -O2 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; get_global_id and its sibling workgroup-query builtins take an integer
; dimension index.

; CHECK: LLVM ERROR: Expect an integer <Dimindx> argument

declare spir_func i64 @_Z13get_global_idj(bfloat)

define spir_kernel void @fuzz_kernel(ptr addrspace(1) %out, bfloat %dim) {
  %id = call spir_func i64 @_Z13get_global_idj(bfloat %dim)
  store i64 %id, ptr addrspace(1) %out
  ret void
}
