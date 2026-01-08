; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpDecorate %[[#]] UserSemantic "annotation_on_function"

@.str = private unnamed_addr constant [23 x i8] c"annotation_on_function\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"an.cl\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

define dso_local spir_func void @foo() {
entry:
  ret void
}
