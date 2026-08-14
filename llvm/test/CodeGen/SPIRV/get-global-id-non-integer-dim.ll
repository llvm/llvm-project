; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s

; get_global_id and its sibling workgroup-query builtins take an integer
; dimension index.

; CHECK: %[[#Func:]] = OpFunction %[[#]] None %[[#]]
; CHECK: OpFunctionParameter
; CHECK: OpFunctionEnd
; CHECK: OpFunctionCall %[[#]] %[[#Func]]

declare spir_func i64 @_Z13get_global_idj(bfloat)

define spir_kernel void @fuzz_kernel(ptr addrspace(1) %out, bfloat %dim) {
  %id = call spir_func i64 @_Z13get_global_idj(bfloat %dim)
  store i64 %id, ptr addrspace(1) %out
  ret void
}
