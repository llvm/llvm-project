; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - 
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@__const.__assert_fail.fmt = external addrspace(1) constant [47 x i8]

define spir_func void @__assertfail() addrspace(4) {
entry:
  ret void
}

define spir_kernel void @_Z3fooPi() addrspace(4) {
entry:
  %input.addr = alloca ptr addrspace(4), align 8
  ret void
}