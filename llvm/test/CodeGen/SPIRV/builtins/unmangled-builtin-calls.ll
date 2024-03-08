; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpDecorate %[[#Id:]] BuiltIn GlobalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[#Id:]] BuiltIn GlobalLinearId
; CHECK-SPIRV:     %[[#Id:]] = OpVariable %[[#]]
; CHECK-SPIRV:     %[[#Id:]] = OpVariable %[[#]]

define spir_kernel void @f() {
entry:
  %call1 = call spir_func i32 @__spirv_BuiltInGlobalLinearId()
  %call2 = call spir_func i64 @__spirv_BuiltInGlobalInvocationId(i32 1)
  ret void
}

declare spir_func i32 @__spirv_BuiltInGlobalLinearId()
declare spir_func i64 @__spirv_BuiltInGlobalInvocationId(i32)
