; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.Inner = type { i32, <4 x float> }
%struct.Outer = type { [5 x %struct.Inner], i32 }

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Inner:]] = OpTypeStruct %[[#int]] %[[#vec4]]
; CHECK-DAG: %[[#idx_5:]] = OpConstant %[[#int]] 5
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Inner]] %[[#idx_5]]
; CHECK-DAG: %[[#Outer:]] = OpTypeStruct %[[#Array]] %[[#int]]
; CHECK-DAG: %[[#ptr_Outer:]] = OpTypePointer Function %[[#Outer]]
; CHECK-DAG: %[[#ptr_float:]] = OpTypePointer Function %[[#float]]

define spir_func float @nested_access(ptr %obj) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#obj_var:]] = OpFunctionParameter %[[#ptr_Outer]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.Outer) %obj, i32 0, i32 2, i32 1, i32 1)
  ; CHECK: %[[#ptr_elem:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#obj_var]] %[[#]] %[[#]] %[[#]] %[[#]]

  %2 = load float, ptr %1, align 4
  ; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr_elem]] Aligned 4

  ret float %2
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
