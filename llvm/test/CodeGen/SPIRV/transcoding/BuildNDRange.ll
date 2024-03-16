; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#]] = OpBuildNDRange %[[#]] %[[#GWS:]] %[[#LWS:]] %[[#GWO:]]
; CHECK-SPIRV-DAG: %[[#GWS]] = OpConstant %[[#]] 123
; CHECK-SPIRV-DAG: %[[#LWS]] = OpConstant %[[#]] 456
; CHECK-SPIRV-DAG: %[[#GWO]] = OpConstant %[[#]] 0

%struct.ndrange_t = type { i32, [3 x i32], [3 x i32], [3 x i32] }

define spir_kernel void @test() {
  %ndrange = alloca %struct.ndrange_t, align 4
  call spir_func void @_Z10ndrange_1Djj(%struct.ndrange_t* sret(%struct.ndrange_t*) %ndrange, i32 123, i32 456)
  ret void
}

declare spir_func void @_Z10ndrange_1Djj(%struct.ndrange_t* sret(%struct.ndrange_t*), i32, i32)
