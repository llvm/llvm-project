; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - 2>&1 | FileCheck %s

@glob = addrspace(1) global ptr null, align 8

; Function Attrs: mustprogress noinline norecurse optnone
define noundef i32 @main() {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 0, ptr addrspace(4) %retval.ascast, align 4
  store ptr @_Z3fooiz, ptr addrspace(4) addrspacecast (ptr addrspace(1) @glob to ptr addrspace(4)), align 8
  call spir_func void (i32, ...) @_Z3fooiz(i32 noundef 5, i32 noundef 3)
  ret i32 0
}

; CHECK: SPIR-V does not support variadic functions
declare spir_func void @_Z3fooiz(i32 noundef, ...)
