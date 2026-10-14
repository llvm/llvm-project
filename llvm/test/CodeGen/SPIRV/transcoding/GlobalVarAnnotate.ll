; Test that llvm.global.annotations on both functions and global variables
; are translated to OpDecorate UserSemantic in SPIR-V.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#FUNC:]] UserSemantic "annotation_on_function"
; CHECK-DAG: OpDecorate %[[#GVAR:]] UserSemantic "annotation_on_global_var"
; CHECK-DAG: %[[#FUNC]] = OpFunction
; CHECK-DAG: %[[#GVAR]] = OpVariable

@.str = private unnamed_addr constant [23 x i8] c"annotation_on_function\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"an.cl\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [25 x i8] c"annotation_on_global_var\00", section "llvm.metadata"
@g = addrspace(1) global i32 0, align 4
@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @.str, ptr @.str.1, i32 2, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr addrspacecast (ptr addrspace(1) @g to ptr), ptr @.str.2, ptr @.str.1, i32 3, ptr null }], section "llvm.metadata"

define dso_local spir_func void @foo() {
entry:
  ret void
}
