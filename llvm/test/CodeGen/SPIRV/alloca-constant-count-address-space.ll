; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_untyped_pointers %s -o - 2>&1 | FileCheck %s

; Generic-storage allocas must not be folded into Function-storage variables.
; CHECK: LLVM ERROR: array allocation: this instruction requires the following SPIR-V extension: SPV_INTEL_variable_length_array
define spir_func void @generic_constant_count() {
entry:
  %array = alloca i32, i32 4, align 4, addrspace(4)
  store volatile i32 1, ptr addrspace(4) %array, align 4
  ret void
}
