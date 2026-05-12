;; This test checks if we generate a single builtin variable for the following
;; LLVM IR.
;; @__spirv_BuiltInLocalInvocationId - A global variable
;; %3 = tail call i64 @_Z12get_local_idj(i32 0) - A function call

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#]] "__spirv_BuiltInLocalInvocationId"
; CHECK-NOT: OpName %[[#]] "__spirv_BuiltInLocalInvocationId.1"

@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

declare spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr

define spir_kernel void @test(i32 %a) {
entry:
  %builtin_call = tail call i64 @_Z12get_local_idj(i32 0)
  ret void
}
