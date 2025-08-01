; Ensure that the backend satisfies the requirement of the verifier
; that disallows uses of intrinsic global variables.

; int *ptr_0 = nullptr;
; void *ptr_1 = ptr_0;
; clang -S -emit-llvm --target=spir example.cpp

; Test passes if use of "-verify-machineinstrs" doesn't lead to crash.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; CHECK: OpFunction

@ptr_0 = dso_local global ptr null, align 4
@ptr_1 = dso_local global ptr null, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_example.cpp, ptr null }]

define internal spir_func void @__cxx_global_var_init() {
entry:
  %0 = load ptr, ptr @ptr_0, align 4
  store ptr %0, ptr @ptr_1, align 4
  ret void
}

define internal spir_func void @_GLOBAL__sub_I_example.cpp() {
entry:
  call spir_func void @__cxx_global_var_init()
  ret void
}
