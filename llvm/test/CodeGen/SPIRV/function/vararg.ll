; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-vulkan --spirv-ext=+SPV_INTEL_function_pointers < %s 2>&1 | FileCheck %s

define void @bar() {
entry:
  call spir_func void (i32, ...) @_Z3fooiz(i32 5, i32 3)
  ret void
}

; CHECK:error: {{.*}} in function bar void (): SPIR-V shaders do not support variadic functions
declare spir_func void @_Z3fooiz(i32, ...)
