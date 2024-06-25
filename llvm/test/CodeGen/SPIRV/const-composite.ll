; This test is to ensure that OpConstantComposite reuses a constant when it's
; already created and available in the same machine function. In this test case
; it's `1` that is passed implicitly as a part of the `foo` function argument
; and also takes part in a composite constant creation.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#type_int32:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#const1:]] = OpConstant %[[#type_int32]] 1
; CHECK-SPIRV: OpTypeArray %[[#]] %[[#const1:]]
; CHECK-SPIRV: %[[#const0:]] = OpConstant %[[#type_int32]] 0
; CHECK-SPIRV: OpConstantComposite %[[#]] %[[#const0]] %[[#const1]]

%struct = type { [1 x i64] }

define spir_kernel void @foo(ptr noundef byval(%struct) %arg) {
entry:
  call spir_func void @bar(<2 x i32> noundef <i32 0, i32 1>)
  ret void
}

define spir_func void @bar(<2 x i32> noundef) {
entry:
  ret void
}
